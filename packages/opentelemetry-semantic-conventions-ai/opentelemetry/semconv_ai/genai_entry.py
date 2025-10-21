"""
GenAI Entry Detection Utilities
===============================

Common utilities for detecting and marking GenAI entry spans across all
GenAI instrumentation packages with comprehensive error handling and fallback mechanisms.
"""

import os
import logging
import threading
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Environment variable for controlling GenAI entry marking
GENAI_ENTRY_ENV_VAR = "OTEL_MARK_GENAI_ENTRY"

# Environment variable for safe mode (disable on errors)
GENAI_ENTRY_SAFE_MODE_VAR = "OTEL_GENAI_ENTRY_SAFE_MODE"

# Span attribute name for marking GenAI entry spans
GENAI_ENTRY_ATTRIBUTE = "gen_ai.is_entry"

# Thread-local storage for tracking GenAI operation nesting depth
_thread_local = threading.local()

# Thread-local storage for recursion detection
_recursion_guard = threading.local()

# Circuit breaker for error handling
_circuit_breaker = {
    'failures': 0,
    'last_failure_time': 0,
    'is_open': False,
    'recovery_timeout': 60,  # 60 seconds
    'failure_threshold': 5   # 5 consecutive failures
}
_circuit_breaker_lock = threading.RLock()


class GenAIEntryDetectionError(Exception):
    """Custom exception for GenAI entry detection errors."""
    pass


def _is_safe_mode_enabled() -> bool:
    """Check if safe mode is enabled to disable entry detection on errors."""
    return os.getenv(GENAI_ENTRY_SAFE_MODE_VAR, "false").lower() == "true"


def _check_circuit_breaker() -> bool:
    """
    Check if circuit breaker allows operation.
    Returns True if operation is allowed, False if circuit is open.
    """
    try:
        with _circuit_breaker_lock:
            current_time = time.time()

            # Check if we should attempt recovery
            if (_circuit_breaker['is_open'] and
                    current_time - _circuit_breaker['last_failure_time'] > _circuit_breaker['recovery_timeout']):
                _circuit_breaker['is_open'] = False
                _circuit_breaker['failures'] = 0
                logger.info("GenAI entry detection circuit breaker: attempting recovery")

            return not _circuit_breaker['is_open']
    except Exception as e:
        logger.debug(f"Circuit breaker check failed: {e}")
        return True  # Default to allow operation


def _record_circuit_breaker_failure():
    """Record a failure in the circuit breaker."""
    try:
        with _circuit_breaker_lock:
            _circuit_breaker['failures'] += 1
            _circuit_breaker['last_failure_time'] = time.time()

            if _circuit_breaker['failures'] >= _circuit_breaker['failure_threshold']:
                _circuit_breaker['is_open'] = True
                logger.warning(
                    f"GenAI entry detection circuit breaker: opened after "
                    f"{_circuit_breaker['failures']} failures. "
                    f"Entry detection disabled for {_circuit_breaker['recovery_timeout']} seconds."
                )
    except Exception as e:
        logger.debug(f"Failed to record circuit breaker failure: {e}")


def _record_circuit_breaker_success():
    """Record a successful operation in the circuit breaker."""
    try:
        with _circuit_breaker_lock:
            if _circuit_breaker['failures'] > 0:
                _circuit_breaker['failures'] = 0
                logger.info("GenAI entry detection circuit breaker: reset after successful operation")
    except Exception as e:
        logger.debug(f"Failed to record circuit breaker success: {e}")


def is_genai_entry_enabled() -> bool:
    """
    Check if GenAI entry marking is enabled via environment variable.

    Returns:
        bool: True if enabled (default), False if disabled or in safe mode
    """
    try:
        if _is_safe_mode_enabled():
            return False

        if not _check_circuit_breaker():
            return False

        return os.getenv(GENAI_ENTRY_ENV_VAR, "true").lower() == "true"
    except Exception as e:
        logger.debug(f"Error checking if GenAI entry is enabled: {e}")
        return False  # Fail-safe default


def _get_genai_depth() -> int:
    """Get the current GenAI operation nesting depth with error handling."""
    try:
        return getattr(_thread_local, 'genai_operation_depth', 0)
    except Exception as e:
        logger.debug(f"Error getting GenAI depth: {e}")
        return 0  # Fail-safe default


def _set_genai_depth(depth: int) -> bool:
    """Set the GenAI operation depth with error handling."""
    try:
        _thread_local.genai_operation_depth = max(0, depth)
        return True
    except Exception as e:
        logger.debug(f"Error setting GenAI depth: {e}")
        return False


def _increment_genai_depth() -> int:
    """Increment and return the GenAI operation depth with error handling."""
    try:
        current_depth = _get_genai_depth()
        new_depth = current_depth + 1
        if _set_genai_depth(new_depth):
            return new_depth
        else:
            return current_depth + 1  # Fallback calculation
    except Exception as e:
        logger.debug(f"Error incrementing GenAI depth: {e}")
        return 1  # Fail-safe default


def _decrement_genai_depth() -> int:
    """Decrement and return the GenAI operation depth with error handling."""
    try:
        current_depth = _get_genai_depth()
        new_depth = max(0, current_depth - 1)
        if _set_genai_depth(new_depth):
            return new_depth
        else:
            return max(0, current_depth - 1)  # Fallback calculation
    except Exception as e:
        logger.debug(f"Error decrementing GenAI depth: {e}")
        return 0  # Fail-safe default


def _is_in_genai_operation() -> bool:
    """
    Check if we're already within a GenAI operation using thread-local storage.
    This is an O(1) operation with comprehensive error handling.

    Returns:
        bool: True if already in GenAI operation, False otherwise
    """
    try:
        return _get_genai_depth() > 0
    except Exception as e:
        logger.debug(f"Error checking GenAI operation state: {e}")
        return False  # Fail-safe default


def _reset_thread_local_state():
    """Reset thread-local state in case of corruption."""
    try:
        _thread_local.genai_operation_depth = 0
        logger.debug("Reset GenAI thread-local state")
    except Exception as e:
        logger.debug(f"Failed to reset thread-local state: {e}")


def _set_genai_operation_state(active: bool) -> None:
    """Set the GenAI operation state in thread-local storage."""
    try:
        _thread_local.genai_operation_active = active
    except Exception as e:
        logger.debug(f"Error setting GenAI thread-local state: {e}")


def _is_in_recursion() -> bool:
    """Check if we're already in a span creation recursion."""
    try:
        return getattr(_recursion_guard, 'in_span_creation', False)
    except Exception:
        return False


def _set_recursion_guard(active: bool) -> None:
    """Set recursion guard state."""
    try:
        _recursion_guard.in_span_creation = active
    except Exception:
        pass


def _find_original_start_span(tracer):
    """
    Try to find the truly original start_span method, bypassing all wrappers.
    This helps avoid calling wrapped versions that could cause recursion.
    """
    try:
        # Try to get the original method from the class
        if hasattr(tracer, '__class__'):
            original_method = getattr(tracer.__class__, 'start_span', None)
            if original_method and callable(original_method):
                return original_method
        
        # Fallback: return the current method
        return getattr(tracer, 'start_span', None)
    except Exception:
        return None


def mark_span_as_genai_entry(span) -> None:
    """
    Mark a span as GenAI entry point only if it's not nested within another GenAI operation.
    Uses efficient O(1) thread-local checking with comprehensive error handling.

    Args:
        span: The OpenTelemetry span to mark
    """
    try:
        if not (span and span.is_recording()):
            return

        if not is_genai_entry_enabled():
            return

        # Only mark as entry if not already in a GenAI operation
        if not _is_in_genai_operation():
            span.set_attribute(GENAI_ENTRY_ATTRIBUTE, True)
            _record_circuit_breaker_success()

    except Exception as e:
        logger.debug(f"Error marking span as GenAI entry: {e}")
        _record_circuit_breaker_failure()
        # Continue execution without raising - fail-safe behavior


def _safe_span_interceptor(original_start_span, expected_depth: int):
    """Create a safe span interceptor with error isolation and recursion detection."""
    def enhanced_start_span(*span_args, **span_kwargs):
        # CRITICAL: Recursion detection - prevent infinite loops
        if _is_in_recursion():
            # If we detect recursion, return a non-recording span to break the cycle
            try:
                from opentelemetry.trace import NonRecordingSpan
                return NonRecordingSpan()
            except Exception:
                # Last resort: try to find and call the truly original method
                try:
                    tracer = span_args[0] if span_args else None
                    if tracer:
                        original_method = _find_original_start_span(tracer)
                        if original_method and original_method != enhanced_start_span:
                            return original_method(tracer, *span_args[1:], **span_kwargs)
                except Exception:
                    pass
                # Final fallback: return non-recording span
                try:
                    from opentelemetry.trace import NonRecordingSpan
                    return NonRecordingSpan()
                except Exception:
                    return None

        span = None
        try:
            # Set recursion guard BEFORE calling original_start_span
            _set_recursion_guard(True)
            
            # Always create the span first
            span = original_start_span(*span_args, **span_kwargs)

            # Try to add entry detection, but don't fail if it errors
            try:
                # Check the REAL-TIME depth, not the expected depth from closure
                current_depth = _get_genai_depth()
                if is_genai_entry_enabled() and current_depth == 1:
                    span.set_attribute(GENAI_ENTRY_ATTRIBUTE, True)
                    _record_circuit_breaker_success()
            except Exception as entry_error:
                # Use print instead of logger to avoid potential recursion in logging
                try:
                    print(f"DEBUG: Error in GenAI entry detection: {entry_error}")
                except Exception:
                    pass  # Even print can fail in extreme cases
                _record_circuit_breaker_failure()
                # Continue without entry detection

        except Exception as span_error:
            # Use print instead of logger.error to avoid recursion in logging
            try:
                print(f"ERROR: Critical error in span creation: {span_error}")
            except Exception:
                pass  # Even print can fail in extreme cases
            # Re-raise span creation errors as they are critical
            raise
        finally:
            # ALWAYS clear recursion guard
            _set_recursion_guard(False)

        return span

    return enhanced_start_span


def with_genai_entry_detection(wrapper_func):
    """
    Decorator that adds GenAI entry detection to wrapper functions with comprehensive error handling.

    This decorator implements multiple layers of error protection:
    1. Exception isolation - errors don't propagate to user code
    2. Circuit breaker - automatic disable on repeated failures
    3. Graceful degradation - continue normal operation if entry detection fails
    4. Recovery mechanisms - automatic re-enable after timeout

    Args:
        wrapper_func: The wrapper function to enhance

    Returns:
        Enhanced wrapper function with GenAI entry detection and error protection
    """
    import asyncio

    if asyncio.iscoroutinefunction(wrapper_func):
        @wraps(wrapper_func)
        async def enhanced_async_wrapper(*args, **kwargs):
            # Early exit if entry detection is disabled
            if not is_genai_entry_enabled():
                return await wrapper_func(*args, **kwargs)

            depth = None
            original_start_span = None
            tracer = None

            try:
                # Increment depth when entering GenAI operation
                depth = _increment_genai_depth()

                # Store the original span creation to intercept it
                if len(args) > 0 and hasattr(args[0], 'start_span'):
                    tracer = args[0]
                    original_start_span = tracer.start_span
                    tracer.start_span = _safe_span_interceptor(original_start_span, depth)

                # Call the original wrapper function
                result = await wrapper_func(*args, **kwargs)
                return result

            except Exception as e:
                logger.debug(f"Error in GenAI entry detection async wrapper: {e}")
                _record_circuit_breaker_failure()
                # Always continue with original function to avoid breaking user code
                try:
                    if original_start_span is not None and tracer is not None:
                        tracer.start_span = original_start_span
                    return await wrapper_func(*args, **kwargs)
                except Exception:
                    raise  # Re-raise the original exception

            finally:
                # Always cleanup, even if errors occurred
                try:
                    if original_start_span is not None and tracer is not None:
                        tracer.start_span = original_start_span
                except Exception as cleanup_error:
                    logger.debug(f"Error restoring original start_span: {cleanup_error}")

                try:
                    if depth is not None:
                        _decrement_genai_depth()
                except Exception as depth_error:
                    logger.debug(f"Error decrementing depth: {depth_error}")
                    # Attempt to reset state on persistent errors
                    _reset_thread_local_state()

        return enhanced_async_wrapper
    else:
        @wraps(wrapper_func)
        def enhanced_wrapper(*args, **kwargs):
            # Early exit if entry detection is disabled
            if not is_genai_entry_enabled():
                return wrapper_func(*args, **kwargs)

            depth = None
            original_start_span = None
            tracer = None

            try:
                # Increment depth when entering GenAI operation
                depth = _increment_genai_depth()

                # Store the original span creation to intercept it
                if len(args) > 0 and hasattr(args[0], 'start_span'):
                    tracer = args[0]
                    original_start_span = tracer.start_span
                    
                    # Check if we've already wrapped this tracer to avoid double-wrapping
                    if not hasattr(original_start_span, '_genai_entry_wrapped'):
                        enhanced_span_func = _safe_span_interceptor(original_start_span, depth)
                        # Mark the enhanced function to prevent double-wrapping
                        enhanced_span_func._genai_entry_wrapped = True
                        enhanced_span_func._original_start_span = original_start_span
                        tracer.start_span = enhanced_span_func
                    else:
                        # Already wrapped, use the original function stored in the wrapper
                        original_start_span = getattr(original_start_span, '_original_start_span', original_start_span)

                # Call the original wrapper function
                result = wrapper_func(*args, **kwargs)
                return result

            except Exception as e:
                # Use print instead of logger to avoid potential recursion
                try:
                    print(f"DEBUG: Error in GenAI entry detection wrapper: {e}")
                except Exception:
                    pass  # Even print can fail in extreme cases
                _record_circuit_breaker_failure()
                # Always continue with original function to avoid breaking user code
                try:
                    if original_start_span is not None and tracer is not None:
                        # If we wrapped the tracer, restore the original method
                        if hasattr(tracer.start_span, '_genai_entry_wrapped'):
                            tracer.start_span = getattr(tracer.start_span, '_original_start_span', original_start_span)
                        else:
                            tracer.start_span = original_start_span
                    return wrapper_func(*args, **kwargs)
                except Exception:
                    raise  # Re-raise the original exception

            finally:
                # Always cleanup, even if errors occurred
                try:
                    if original_start_span is not None and tracer is not None:
                        # If we wrapped the tracer, restore the original method
                        if hasattr(tracer.start_span, '_genai_entry_wrapped'):
                            tracer.start_span = getattr(tracer.start_span, '_original_start_span', original_start_span)
                        else:
                            tracer.start_span = original_start_span
                except Exception as cleanup_error:
                    # Use print instead of logger to avoid potential recursion
                    try:
                        print(f"DEBUG: Error restoring original start_span: {cleanup_error}")
                    except Exception:
                        pass  # Even print can fail in extreme cases

                try:
                    if depth is not None:
                        _decrement_genai_depth()
                except Exception as depth_error:
                    # Use print instead of logger to avoid potential recursion
                    try:
                        print(f"DEBUG: Error decrementing depth: {depth_error}")
                    except Exception:
                        pass  # Even print can fail in extreme cases
                    # Attempt to reset state on persistent errors
                    _reset_thread_local_state()

        return enhanced_wrapper


def get_genai_entry_detection_health() -> dict:
    """
    Get health status of GenAI entry detection system.

    Returns:
        dict: Health status including circuit breaker state, error counts, etc.
    """
    try:
        # Use separate calls to avoid deadlock
        enabled = is_genai_entry_enabled()
        safe_mode = _is_safe_mode_enabled()
        current_depth = _get_genai_depth()

        with _circuit_breaker_lock:
            return {
                'enabled': enabled,
                'safe_mode': safe_mode,
                'circuit_breaker': {
                    'is_open': _circuit_breaker['is_open'],
                    'failures': _circuit_breaker['failures'],
                    'last_failure_time': _circuit_breaker['last_failure_time'],
                    'recovery_timeout': _circuit_breaker['recovery_timeout']
                },
                'current_depth': current_depth
            }
    except Exception as e:
        return {
            'error': str(e),
            'enabled': False
        }

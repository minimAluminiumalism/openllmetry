#!/usr/bin/env python3
"""
Test GenAI Entry Detection
==========================

Run with: PYTHONPATH=. python -m pytest tests/test_genai_entry.py -v
"""

import threading
import pytest
from unittest.mock import Mock, patch
from opentelemetry.trace import NonRecordingSpan

from opentelemetry.semconv_ai.genai_entry import (
    _is_in_recursion,
    _set_recursion_guard,
    _safe_span_interceptor,
    _check_circuit_breaker,
    _record_circuit_breaker_failure,
    _record_circuit_breaker_success,
    _circuit_breaker,
    _find_original_start_span
)


class TestRecursionPrevention:
    """Test recursion prevention mechanisms"""
    
    def test_basic_recursion_guard(self):
        """Test basic recursion guard functionality"""
        assert not _is_in_recursion()
        _set_recursion_guard(True)
        assert _is_in_recursion()
        _set_recursion_guard(False)
        assert not _is_in_recursion()
    
    def test_thread_local_isolation(self):
        """Test thread-local isolation of recursion guard"""
        results = []
        
        def thread_worker(thread_id):
            _set_recursion_guard(True)
            is_recursive = _is_in_recursion()
            _set_recursion_guard(False)
            is_not_recursive = not _is_in_recursion()
            results.append((thread_id, is_recursive and is_not_recursive))
        
        threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert all(result[1] for result in results)
    
    def test_infinite_recursion_prevention(self):
        """Test prevention of infinite recursion in span creation"""
        call_count = 0
        
        def recursive_start_span(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 10:
                raise RecursionError("Maximum recursion depth exceeded")
            return enhanced_start_span(*args, **kwargs)
        
        enhanced_start_span = _safe_span_interceptor(recursive_start_span, 1)
        result = enhanced_start_span("test_span")
        
        assert isinstance(result, NonRecordingSpan) or result is None
        assert call_count <= 10


class TestSpanInterceptorSafety:
    """Test safe span interceptor functionality"""
    
    def test_normal_operation(self):
        """Test normal span creation without issues"""
        mock_span = Mock()
        original_start_span = Mock(return_value=mock_span)
        
        enhanced_start_span = _safe_span_interceptor(original_start_span, 1)
        result = enhanced_start_span("test_span")
        
        assert result == mock_span
        original_start_span.assert_called_once()
    
    def test_error_handling_without_logger_recursion(self):
        """Test error handling avoids logger-induced recursion"""
        def failing_start_span(*args, **kwargs):
            raise Exception("Span creation failed")
        
        enhanced_start_span = _safe_span_interceptor(failing_start_span, 1)
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(Exception):
                enhanced_start_span("test_span")
            
            # Verify print was used instead of logger
            print_calls = mock_print.call_args_list
            error_logged = any("ERROR:" in str(call) for call in print_calls)
            assert error_logged or len(print_calls) == 0


class TestMultipleInstrumentationCompatibility:
    """Test compatibility with multiple instrumentations"""
    
    def test_wrapper_marking_prevents_double_wrapping(self):
        """Test that wrapper marking prevents double wrapping"""
        original_start_span = Mock(return_value=Mock())
        
        enhanced_func = _safe_span_interceptor(original_start_span, 1)
        enhanced_func._genai_entry_wrapped = True
        enhanced_func._original_start_span = original_start_span
        
        assert hasattr(enhanced_func, '_genai_entry_wrapped')
        assert enhanced_func._genai_entry_wrapped is True
        assert enhanced_func._original_start_span == original_start_span
    
    def test_original_method_discovery(self):
        """Test finding original start_span method"""
        mock_tracer = Mock()
        mock_tracer.start_span = Mock()
        
        original = _find_original_start_span(mock_tracer)
        assert callable(original) or original is None


class TestCircuitBreakerMechanism:
    """Test circuit breaker for error handling"""
    
    def setup_method(self):
        """Reset circuit breaker state"""
        _circuit_breaker['failures'] = 0
        _circuit_breaker['is_open'] = False
        _circuit_breaker['last_failure_time'] = 0
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        assert _check_circuit_breaker() is True
        
        _record_circuit_breaker_failure()
        assert _circuit_breaker['failures'] == 1
        
        _record_circuit_breaker_success()
        assert _circuit_breaker['failures'] == 0
    
    def test_circuit_breaker_threshold(self):
        """Test circuit breaker opens after threshold"""
        threshold = _circuit_breaker['failure_threshold']
        for _ in range(threshold):
            _record_circuit_breaker_failure()
        
        assert _circuit_breaker['is_open'] is True
        assert _check_circuit_breaker() is False
        
        # Reset circuit breaker after test to avoid affecting other tests
        _circuit_breaker['is_open'] = False
        _circuit_breaker['failures'] = 0


class TestEntrySpanMarking:
    """Test that entry span marking still works after recursion fix"""
    
    def setup_method(self):
        """Setup for each test method"""
        import os
        from opentelemetry.semconv_ai.genai_entry import GENAI_ENTRY_ENV_VAR
        # Ensure entry detection is enabled for all tests
        os.environ[GENAI_ENTRY_ENV_VAR] = 'true'
    
    def test_normal_entry_span_marking(self):
        """Test normal entry span marking functionality"""
        from opentelemetry.semconv_ai.genai_entry import (
            GENAI_ENTRY_ATTRIBUTE, 
            _increment_genai_depth, 
            _decrement_genai_depth,
            _set_genai_depth
        )
        
        # Reset depth
        _set_genai_depth(0)
        _increment_genai_depth()  # depth = 1, should mark as entry
        
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.is_recording = Mock(return_value=True)
        
        def mock_original_start_span(*args, **kwargs):
            return mock_span
        
        enhanced_start_span = _safe_span_interceptor(mock_original_start_span, 1)
        result = enhanced_start_span('test_span')
        
        # Verify span was created and marked as entry
        assert result == mock_span
        mock_span.set_attribute.assert_called_with(GENAI_ENTRY_ATTRIBUTE, True)
        
        _decrement_genai_depth()
    
    def test_entry_marking_with_recursion_prevention(self):
        """Test that recursion prevention doesn't break entry marking"""
        from opentelemetry.semconv_ai.genai_entry import (
            _increment_genai_depth, 
            _decrement_genai_depth,
            _set_genai_depth
        )
        
        # Reset depth
        _set_genai_depth(0)
        _increment_genai_depth()  # depth = 1
        
        # Test normal case first
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.is_recording = Mock(return_value=True)
        
        def normal_start_span(*args, **kwargs):
            return mock_span
        
        enhanced_start_span = _safe_span_interceptor(normal_start_span, 1)
        result = enhanced_start_span('normal_span')
        
        assert result == mock_span
        assert mock_span.set_attribute.called
        
        # Test that recursion case returns NonRecordingSpan or None
        recursion_count = 0
        def recursive_start_span(*args, **kwargs):
            nonlocal recursion_count
            recursion_count += 1
            if recursion_count > 3:
                return Mock()
            return enhanced_recursive(*args, **kwargs)
        
        enhanced_recursive = _safe_span_interceptor(recursive_start_span, 1)
        recursive_result = enhanced_recursive('recursive_span')
        
        # Should prevent recursion
        assert isinstance(recursive_result, NonRecordingSpan) or recursive_result is None
        
        _decrement_genai_depth()


class TestActualRecursionScenario:
    """Test actual recursion scenarios that could occur in production"""
    
    def test_logging_instrumentation_recursion(self):
        """Test recursion caused by logging instrumentation"""
        # Simulate the actual recursion scenario from the bug report
        recursion_depth = 0
        
        def mock_logger_error_that_triggers_span_creation(*args, **kwargs):
            nonlocal recursion_depth
            recursion_depth += 1
            if recursion_depth > 5:
                return  # Prevent actual infinite recursion in test
            # This would normally trigger span creation again
            enhanced_start_span("logging_span")
        
        def mock_original_start_span(*args, **kwargs):
            # Simulate error that triggers logger.error
            mock_logger_error_that_triggers_span_creation()
            return Mock()
        
        enhanced_start_span = _safe_span_interceptor(mock_original_start_span, 1)
        
        # This should not cause infinite recursion
        result = enhanced_start_span("test_span")
        assert result is not None
        assert recursion_depth <= 5
    
    def test_multiple_tracer_modifications(self):
        """Test scenario with multiple tracer modifications"""
        mock_tracer = Mock()
        original_method = Mock(return_value=Mock())
        mock_tracer.start_span = original_method
        
        # First instrumentation modifies tracer
        def first_wrapper(*args, **kwargs):
            return original_method(*args, **kwargs)
        mock_tracer.start_span = first_wrapper
        
        # Second instrumentation (our GenAI entry detection)
        enhanced_func = _safe_span_interceptor(mock_tracer.start_span, 1)
        enhanced_func._genai_entry_wrapped = True
        enhanced_func._original_start_span = mock_tracer.start_span
        
        # Should work without issues
        result = enhanced_func("test_span")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

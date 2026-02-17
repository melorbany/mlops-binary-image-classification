"""Tests for inference functions."""
import pytest
import torch
import io
from PIL import Image
from pathlib import Path
import tempfile

from src.models.cnn import SimpleCNN, get_model

class TestModel:
    """Test suite for model functions."""
    
    def test_simple_cnn_output_shape(self):
        """Test that SimpleCNN outputs correct shape."""
        model = SimpleCNN(num_classes=2)
        
        # Create dummy input (batch_size=4, channels=3, height=224, width=224)
        dummy_input = torch.randn(4, 3, 224, 224)
        
        output = model(dummy_input)
        
        assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    
    def test_simple_cnn_forward_pass(self):
        """Test forward pass doesn't raise errors."""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model("simple_cnn", num_classes=2)
        
        assert isinstance(model, SimpleCNN)
    
    def test_get_model_invalid_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            get_model("nonexistent_model")
    
    def test_model_save_load(self):
        """Test model serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            
            # Create and save model
            model = SimpleCNN(num_classes=2)
            dummy_input = torch.randn(1, 3, 224, 224)
            original_output = model(dummy_input)
            
            torch.save(model.state_dict(), model_path)
            
            # Load model
            loaded_model = SimpleCNN(num_classes=2)
            loaded_model.load_state_dict(torch.load(model_path))
            loaded_output = loaded_model(dummy_input)
            
            assert torch.allclose(original_output, loaded_output)

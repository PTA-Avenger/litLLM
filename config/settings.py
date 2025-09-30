"""Configuration management for the Stylistic Poetry LLM Framework."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model parameters."""
    base_model_name: str = Field(default="gpt2", description="Base model to use for fine-tuning")
    max_length: int = Field(default=512, description="Maximum sequence length")
    temperature: float = Field(default=0.8, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    num_return_sequences: int = Field(default=1, description="Number of sequences to generate")


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    learning_rate: float = Field(default=5e-5, description="Learning rate for fine-tuning")
    batch_size: int = Field(default=8, description="Training batch size")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_steps: int = Field(default=500, description="Number of warmup steps")
    save_steps: int = Field(default=1000, description="Save checkpoint every N steps")
    eval_steps: int = Field(default=500, description="Evaluate every N steps")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")


class DataConfig(BaseModel):
    """Configuration for data processing."""
    data_dir: str = Field(default="data", description="Directory containing poetry data")
    output_dir: str = Field(default="models", description="Directory for model outputs")
    max_poems_per_poet: int = Field(default=1000, description="Maximum poems to use per poet")
    min_line_length: int = Field(default=3, description="Minimum line length in characters")
    max_line_length: int = Field(default=200, description="Maximum line length in characters")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    enable_quantitative: bool = Field(default=True, description="Enable quantitative evaluation")
    enable_qualitative: bool = Field(default=True, description="Enable qualitative evaluation")
    comparison_poets: list[str] = Field(default_factory=list, description="Poets to compare against")
    metrics_output_dir: str = Field(default="results", description="Directory for evaluation results")


class SystemConfig(BaseModel):
    """Main system configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # System settings
    log_level: str = Field(default="INFO", description="Logging level")
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/default.yaml")
        self._config: Optional[SystemConfig] = None
    
    def load_config(self, config_path: Optional[Path] = None) -> SystemConfig:
        """Load configuration from YAML file."""
        if config_path:
            self.config_path = config_path
            
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self._config = SystemConfig(**config_data)
        else:
            # Use default configuration
            self._config = SystemConfig()
            self.save_config()  # Save default config for reference
            
        return self._config
    
    def save_config(self, config: Optional[SystemConfig] = None) -> None:
        """Save configuration to YAML file."""
        if config:
            self._config = config
            
        if self._config is None:
            self._config = SystemConfig()
            
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> SystemConfig:
        """Update configuration with new values."""
        current_config = self.get_config()
        config_dict = current_config.model_dump()
        
        # Deep update the configuration
        self._deep_update(config_dict, updates)
        
        # Validate and create new config
        self._config = SystemConfig(**config_dict)
        return self._config
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global config manager instance
config_manager = ConfigManager()
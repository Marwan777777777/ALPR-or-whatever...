"""Actionable structured domain exceptions for Vehicle Intelligence."""

from typing import Optional


class VehicleIntelligenceError(Exception):
    """Base exception for all vehicle intelligence errors."""
    def __init__(self, message: str, actionable_advice: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.actionable_advice = actionable_advice

    def __str__(self) -> str:
        if self.actionable_advice:
            return f"{self.message} -> Advice: {self.actionable_advice}"
        return self.message


class ModelWeightsNotFoundError(VehicleIntelligenceError):
    """Raised when model weights cannot be found at the specified path."""
    def __init__(self, model_name: str, weights_path: str):
        super().__init__(
            message=f"Model weights for '{model_name}' not found at: '{weights_path}'",
            actionable_advice=(
                f"Verify that model weights are placed at '{weights_path}' or configure "
                "a mock detector for non-GPU development / tests."
            )
        )


class MediaReadError(VehicleIntelligenceError):
    """Raised when an image or video stream cannot be opened or decoded."""
    def __init__(self, source_path: str, reason: str):
        super().__init__(
            message=f"Failed to read media from '{source_path}': {reason}",
            actionable_advice="Verify file existence, codec compatibility, or network RTSP stream availability."
        )


class DeviceUnavailableError(VehicleIntelligenceError):
    """Raised when requested acceleration device (e.g. CUDA) is not available."""
    def __init__(self, requested_device: str, available_devices: str):
        super().__init__(
            message=f"Requested device '{requested_device}' is not available (Found: {available_devices})",
            actionable_advice="Check NVIDIA drivers, install CUDA-enabled PyTorch, or fall back to 'cpu'."
        )


class InvalidConfigurationError(VehicleIntelligenceError):
    """Raised when configuration validation fails."""
    pass

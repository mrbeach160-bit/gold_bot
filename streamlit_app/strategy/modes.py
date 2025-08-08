"""
Entry Mode Configuration

Defines constants and configuration for different smart entry modes.
"""

# Entry mode constants
ENTRY_MODE_MOMENTUM = "momentum"
ENTRY_MODE_PULLBACK = "pullback"

# Available entry modes
ENTRY_MODES = [ENTRY_MODE_MOMENTUM, ENTRY_MODE_PULLBACK]

# Default entry mode (for backward compatibility)
DEFAULT_ENTRY_MODE = ENTRY_MODE_MOMENTUM

# Entry mode display names
ENTRY_MODE_DISPLAY_NAMES = {
    ENTRY_MODE_MOMENTUM: "Momentum",
    ENTRY_MODE_PULLBACK: "Pullback"
}

# Entry mode descriptions
ENTRY_MODE_DESCRIPTIONS = {
    ENTRY_MODE_MOMENTUM: "Traditional momentum-based entry with immediate execution at optimal price levels",
    ENTRY_MODE_PULLBACK: "Wait for price retracement toward support/resistance for better entry opportunities"
}
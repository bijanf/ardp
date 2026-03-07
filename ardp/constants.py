"""Physical and regional constants for AMOC diagnostics."""

# Reference values
S0: float = 35.0  # Reference salinity [PSU]
RHO_0: float = 1025.0  # Reference density [kg/m^3]
CP: float = 3992.0  # Specific heat capacity of seawater [J/(kg·K)]

# Key latitudes
SAMBA_LAT: float = -34.5  # SAMBA array latitude [degrees_north]
RAPID_LAT: float = 26.5   # RAPID array latitude [degrees_north]

# Region bounds (lon_min, lon_max, lat_min, lat_max)
SUBTROPICAL_SOUTH_ATLANTIC: tuple[float, float, float, float] = (-60.0, 20.0, -35.0, -15.0)
SUBTROPICAL_SOUTH_INDOPACIFIC: tuple[float, float, float, float] = (20.0, 290.0, -35.0, -15.0)
NORTH_ATLANTIC_WARMING_HOLE: tuple[float, float, float, float] = (-50.0, -15.0, 45.0, 60.0)
IRMINGER_SEA: tuple[float, float, float, float] = (-44.0, -28.0, 58.0, 66.0)
GULF_STREAM_REGION: tuple[float, float, float, float] = (-80.0, -45.0, 30.0, 45.0)

# F_ovS validation
FOVS_EXPECTED_TREND: float = -1.20e-3  # Sv/year (= -1.20 mSv/yr)
FOVS_TREND_TOLERANCE: float = 0.5e-3  # Sv/year tolerance

# Density bins for density-space MOC
SIGMA0_BINS_MIN: float = 20.0
SIGMA0_BINS_MAX: float = 28.5
SIGMA0_BINS_STEP: float = 0.1

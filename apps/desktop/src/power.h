#pragma once

// Windows 11 may treat a process without a focused window as background work
// and apply "EcoQoS" power throttling: lower clocks and efficiency cores.
// Reconstruction is throughput-bound, so opt out. No-op elsewhere.
void disablePowerThrottling();

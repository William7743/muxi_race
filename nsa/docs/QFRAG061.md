# NSA061 Q fragment with shared probability PV

Parent057 shared simple kernel changes Qs from shared to fragment, retains
128 threads and groups32 PV FullCol. Tests interaction unlike earlier
64-thread Q-fragment path. Could save shared traffic or increase register
pressure/layout cost. Currently affects G16 too; initial g32 suite only,
so no promotion without gating or additional G16 validation.

g32061 seed412 launched after no active job confirmation. No performance
or compile-success claim yet. Retain057.

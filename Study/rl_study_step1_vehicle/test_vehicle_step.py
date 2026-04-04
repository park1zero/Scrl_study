from __future__ import annotations

import math

from vehicle import VehicleParams, VehicleState, step_vehicle, equivalent_rwa_from_sfa, blend_rwa_command


def run_demo() -> None:
    params = VehicleParams()
    state = VehicleState()

    lam = 0.5
    delta_auto_rwa = 0.12  # automation wants a modest left steer
    sfa_driver_cmd = 0.0   # driver initially does nothing

    print("t,x,y,psi,vy,r,delta_rwa,delta_sfa,delta_drv_rwa,delta_cmd")
    for k in range(80):
        # after 1.0 s the driver starts steering the wheel to the right
        if state.t > 1.0:
            sfa_driver_cmd = -1.5

        delta_drv_rwa = equivalent_rwa_from_sfa(state.delta_sfa, params)
        delta_cmd = blend_rwa_command(lam, delta_drv_rwa, delta_auto_rwa)
        state = step_vehicle(state, rwa_cmd=delta_cmd, sfa_cmd=sfa_driver_cmd, params=params)

        if k % 5 == 0:
            print(
                f"{state.t:.2f},{state.x:.2f},{state.y:.2f},{state.psi:.4f},"
                f"{state.vy:.4f},{state.r:.4f},{state.delta_rwa:.4f},"
                f"{state.delta_sfa:.4f},{delta_drv_rwa:.4f},{delta_cmd:.4f}"
            )

    print("\nFinal lateral position y =", round(state.y, 3), "m")
    print("Final yaw psi =", round(state.psi * 180.0 / math.pi, 3), "deg")


if __name__ == "__main__":
    run_demo()

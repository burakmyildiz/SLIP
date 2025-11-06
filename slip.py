
import numpy as np
from scipy.integrate import solve_ivp

def flight_phase_rule(t, state, gravity):
    h, hdot = state
    h_dotdot = -gravity
    return [hdot, h_dotdot]

def stance_phase_rule(t, state, params):
    h, hdot = state
    k, m, h0, g = params["k"], params["m"], params["h0"], params["g"]
    return [hdot, (k/m) * (h0-h) - g]

def main(mass=10.0, spring_length=2.0, gravity=9.81, k_initial=200.0, k_final=400.0, initial_height=3.0, total_time=10.0):
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    if spring_length <= 0:
        raise ValueError(f"spring_length must be positive, got {spring_length}")
    if gravity <= 0:
        raise ValueError(f"gravity must be positive, got {gravity}")
    if k_initial <= 0:
        raise ValueError(f"k_initial must be positive, got {k_initial}")
    if k_final <= 0:
        raise ValueError(f"k_final must be positive, got {k_final}")
    if initial_height < spring_length:
        raise ValueError(f"initial_height ({initial_height}) must be >= spring_length ({spring_length})")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")

    params = {"k": k_initial, "m": mass, "h0": spring_length, "g": gravity}

    h = initial_height
    hdot = 0.0
    t = 0.0
    current_state = [h, hdot]
    all_times = []
    all_states = []
    current_phase = 'flight'

    def touchdown_event(t, state):
        return state[0] - spring_length

    def liftoff_event(t, state):
        return state[0] - spring_length

    def bottom_event(t, state):
        return state[1]

    touchdown_event.terminal = True
    touchdown_event.direction = -1
    liftoff_event.terminal = True
    liftoff_event.direction = 1
    bottom_event.terminal = True
    bottom_event.direction = 1

    while t < total_time:
        if current_phase == 'flight':
            sol = solve_ivp(
                    lambda tt, yy: flight_phase_rule(tt, yy, gravity),
                    [t, total_time],
                    current_state,
                    events=touchdown_event,
                    dense_output=True
                )
            current_phase = 'stance_compression'

        elif current_phase == 'stance_compression':
            sol = solve_ivp(
                    lambda tt, yy: stance_phase_rule(tt, yy, params),
                    [t, total_time],
                    current_state,
                    events=bottom_event,
                    dense_output=True
                )
            print("Hit bottom! Boosting spring stiffness.")
            params["k"] = k_final
            current_phase = 'stance_decompression'

        elif current_phase == 'stance_decompression':
            sol = solve_ivp(
                    lambda tt, yy: stance_phase_rule(tt, yy, params),
                    [t, total_time],
                    current_state,
                    events=liftoff_event,
                    dense_output=True
                )
            params["k"] = k_initial
            current_phase = 'flight'

        plot_times = np.linspace(sol.t[0], sol.t[-1], 30)
        plot_states = sol.sol(plot_times)
        all_times.append(plot_times)
        all_states.append(plot_states)

        current_state = sol.y[:, -1]
        t = sol.t[-1]

        if sol.status == 1:
            print(f"Phase transition at t={t:.2f}, h={current_state[0]:.2f}, hdot={current_state[1]:.2f}, switching to {current_phase} phase.")
            pass
        else:
            print("Simulation complete.")
            break
    
    finished_times = np.concatenate(all_times)
    finished_states = np.hstack(all_states).T

    return finished_times, finished_states

if __name__ == "__main__":
    from visualize import animate_slip, plot_phase_space

    m = 10.0
    h0 = 4.0
    g = 9.81
    h_max = 5.0
    k_initial = 200.0
    k_final = 400.0

    times, states = main(
        mass=m,
        spring_length=h0,
        gravity=g,
        k_initial=k_initial,
        k_final=k_final,
        initial_height=h_max
    )

    mode = input("Choose visualization mode (animate/phase/both): ").strip().lower()

    save_file = None
    if mode in ["animate", "both"]:
        save_choice = input("Save animation? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_file = input("Enter filename (e.g., slip.mp4): ").strip()

    if mode == "animate":
        animate_slip(times, states, h0, realtime_factor=1.0, save_file=save_file)
    elif mode == "phase":
        plot_phase_space(times, states)
    elif mode == "both":
        plot_phase_space(times, states)
        animate_slip(times, states, h0, realtime_factor=1.0, save_file=save_file)
    else:
        print("Invalid mode. Defaulting to animation.")
        animate_slip(times, states, h0, realtime_factor=1.0, save_file=save_file)

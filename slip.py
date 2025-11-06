
import numpy as np
from scipy.integrate import solve_ivp

g = 9.81
m = 0.5
k = 1
h0 = 1.0
h_max = 1.5
import matplotlib.pyplot as plt

def flight_phase_rule(t, state):
    h, hdot = state
    h_dotdot = -g
    return [hdot, h_dotdot]

def stance_phase_rule(t, state):
    h, hdot = state
    h_dotdot = (k/m) * (h0-h) - g
    return [hdot, h_dotdot]

def touchdown_detection_event(t, state):
    h, hdot = state
    return h - h0

def liftoff_detection_event(t, state):
    h, hdot = state
    return h - h0

touchdown_detection_event.terminal = True
touchdown_detection_event.direction = -1

liftoff_detection_event.terminal = True
liftoff_detection_event.direction = 1

###############################

total_time = 10.0


def main():
    h = h_max
    hdot = 0.0
    t = 0.0
    current_state = [h, hdot]
    all_times = []
    all_states = []
    current_phase = 'flight'

    while t < total_time:
        if current_phase == 'flight':
            sol = solve_ivp(flight_phase_rule, [t, total_time], current_state, events=touchdown_detection_event, dense_output=True)
            current_phase = 'stance'
        elif current_phase == 'stance':
            sol = solve_ivp(stance_phase_rule, [t, total_time], current_state, events=liftoff_detection_event, dense_output=True)
            current_phase = 'flight'

        plot_times = np.linspace(sol.t[0], sol.t[-1], 100)
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
    plt.plot(finished_states[:, 0], finished_states[:, 1])
    plt.xlabel("Position (h)")
    plt.ylabel("Velocity (h_dot)")
    plt.title("SLIP")
    plt.grid(True)
    plt.show()

    return

if __name__ == "__main__":
    main()
    
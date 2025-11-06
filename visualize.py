import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def draw_spring(ax, x0, y0, x1, y1, n_coils=10, width=0.1):
    L = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    theta = np.arctan2(y1-y0, x1-x0)

    spring_x = np.linspace(0, L, n_coils*4)
    spring_y = np.zeros_like(spring_x)

    for i in range(1, n_coils*4-1):
        if i % 2 == 1:
            spring_y[i] = width * (1 if (i//2) % 2 == 0 else -1)

    spring_x_rot = x0 + spring_x * np.cos(theta) - spring_y * np.sin(theta)
    spring_y_rot = y0 + spring_x * np.sin(theta) + spring_y * np.cos(theta)

    ax.plot(spring_x_rot, spring_y_rot, 'b-', linewidth=2)

def animate_slip(times, states, h0, realtime_factor=1.0, save_file=None, h_max=None, show=True):
    if not isinstance(times, np.ndarray) or times.ndim != 1:
        raise ValueError(f"times must be 1D numpy array, got shape {times.shape if isinstance(times, np.ndarray) else type(times)}")
    if not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != 2:
        raise ValueError(f"states must be 2D numpy array with shape (n, 2), got shape {states.shape if isinstance(states, np.ndarray) else type(states)}")
    if len(times) != len(states):
        raise ValueError(f"times and states must have same length, got {len(times)} vs {len(states)}")
    if len(times) < 2:
        raise ValueError(f"Need at least 2 data points for animation, got {len(times)}")

    max_frames = 300
    if len(times) > max_frames:
        step = len(times) // max_frames
        times = times[::step]
        states = states[::step]
        print(f"Decimated to {len(times)} frames (keeping every {step}th frame)")

    if h_max is None:
        h_max = np.max(states[:, 0])

    if save_file:
        fig_size = (12, 5)
    else:
        fig_size = (14, 6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(0, h_max + 0.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('h')
    ax1.set_title('SLIP Animation')
    ax1.grid(True)

    mass_circle, = ax1.plot([], [], 'ro', markersize=20)
    ground_line, = ax1.plot([-0.5, 0.5], [0, 0], 'k-', linewidth=3)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

    ax2.set_xlabel('Position (h)')
    ax2.set_ylabel('Velocity (h_dot)')
    ax2.set_title('Phase Plot')
    ax2.grid(True)
    ax2.plot(states[:, 0], states[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    phase_point, = ax2.plot([], [], 'ro', markersize=8)

    def init():
        mass_circle.set_data([], [])
        phase_point.set_data([], [])
        time_text.set_text('')
        return mass_circle, phase_point, time_text

    def update(frame):
        if frame >= len(times):
            frame = len(times) - 1

        h = states[frame, 0]
        t = times[frame]

        mass_circle.set_data([0], [h])
        phase_point.set_data([h], [states[frame, 1]])
        time_text.set_text(f't = {t:.2f}s\nh = {h:.2f}m')

        for artist in ax1.lines[2:]:
            artist.remove()

        if h <= h0:
            draw_spring(ax1, 0, 0, 0, h, n_coils=15, width=0.08)

        return mass_circle, phase_point, time_text

    dt = np.mean(np.diff(times))
    interval = dt * 1000 / realtime_factor
    fps = min(30, max(20, int(1000/interval)))

    anim = FuncAnimation(fig, update, init_func=init, frames=len(times),
                         interval=interval, blit=False, repeat=True)
    plt.tight_layout()

    if save_file:
        n_frames = len(times)
        print(f"Saving {n_frames} frames to {save_file}...")
        print(f"FPS: {fps}, Duration: {times[-1]-times[0]:.1f}s")

        try:
            if save_file.endswith('.gif'):
                writer = PillowWriter(fps=fps)
                print("Using GIF format (slow). Consider using .mp4 for faster saves.")
            else:
                writer = FFMpegWriter(fps=fps, bitrate=4000)

            class ProgressWriter:
                def __init__(self, base_writer):
                    self.base_writer = base_writer
                    self.frame_count = 0
                    self.total_frames = n_frames

                def __getattr__(self, name):
                    return getattr(self.base_writer, name)

                def grab_frame(self, **kwargs):
                    self.base_writer.grab_frame(**kwargs)
                    self.frame_count += 1
                    if self.frame_count % 50 == 0 or self.frame_count == self.total_frames:
                        print(f"  Progress: {self.frame_count}/{self.total_frames} frames", end='\r')

            progress_writer = ProgressWriter(writer)
            progress_writer.setup = writer.setup
            progress_writer.saving = writer.saving
            progress_writer.finish = writer.finish

            anim.save(save_file, writer=progress_writer, dpi=120)
            print(f"\nAnimation saved successfully!")

        except Exception as e:
            if 'ffmpeg' in str(e).lower():
                print(f"\nError: ffmpeg not found. Install with: sudo apt install ffmpeg")
                print(f"Or save as GIF instead (slower): {save_file.replace('.mp4', '.gif')}")
            raise

    if show:
        plt.show()
    else:
        plt.close()

    return anim

def plot_phase_space(times, states):
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel("Position (h)")
    plt.ylabel("Velocity (h_dot)")
    plt.title("SLIP Phase Plot")
    plt.grid(True)
    plt.show()

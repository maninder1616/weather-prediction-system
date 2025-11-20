from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from probabilities_in_the_sky.models.markov_chain import MarkovChain
    from probabilities_in_the_sky.simulation.simulator import simulate_markov_chain
    from probabilities_in_the_sky.viz.visualize import (
        plot_trajectory,
        plot_distribution_over_time,
        plot_transition_heatmap,
    )
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from probabilities_in_the_sky.models.markov_chain import MarkovChain
    from probabilities_in_the_sky.simulation.simulator import simulate_markov_chain
    from probabilities_in_the_sky.viz.visualize import (
        plot_trajectory,
        plot_distribution_over_time,
        plot_transition_heatmap,
    )


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("☁️ Probabilities in the Sky — Markov Chain Weather Simulator")
        self.geometry("1200x800")
        self.minsize(900, 600)

        # Enhanced ttk style
        try:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            
            # Configure styles for better aesthetics
            style.configure("TLabel", padding=(4, 2), font=("Segoe UI", 9))
            style.configure("TButton", padding=(8, 6), font=("Segoe UI", 9, "bold"))
            style.configure("TButton", background="#1f77b4", foreground="white")
            style.map("TButton",
                     background=[("active", "#1565a0"), ("pressed", "#0d4d7a")])
            style.configure("TLabelframe", padding=(10, 8), font=("Segoe UI", 10, "bold"))
            style.configure("TLabelframe.Label", padding=(4, 2), font=("Segoe UI", 10, "bold"))
            style.configure("TEntry", padding=(4, 4), font=("Segoe UI", 9))
            style.configure("TCombobox", padding=(4, 4), font=("Segoe UI", 9))
            style.configure("TSpinbox", padding=(4, 4), font=("Segoe UI", 9))
        except Exception:
            pass

        self.states: list[str] = ["Sunny", "Cloudy", "Rainy"]
        self.num_states_var = tk.IntVar(value=len(self.states))
        self.days_var = tk.IntVar(value=60)
        self.seed_var = tk.IntVar(value=0)
        self.use_seed_var = tk.BooleanVar(value=False)

        self.matrix_entries: list[list[tk.Entry]] = []

        # variables for UI
        self.initial_state_var = tk.StringVar(value=self.states[0])
        self.status_var = tk.StringVar(value="Ready")

        # root grid weights
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        self._build_controls()
        self._build_plots()
        self._rebuild_matrix_table()

    def _build_controls(self) -> None:
        frame = ttk.LabelFrame(self, text="⚙️ Configuration")
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frame.columnconfigure(7, weight=1)

        num_states_label = ttk.Label(frame, text="Number of states:")
        num_states_label.grid(row=0, column=0, sticky="w", padx=4, pady=4)
        _Tooltip(num_states_label, "Select the number of weather states to model (2-8)")
        
        num_states_spin = ttk.Spinbox(frame, from_=2, to=8, textvariable=self.num_states_var, width=5, command=self._on_states_changed)
        num_states_spin.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        _Tooltip(num_states_spin, "Number of states in the Markov chain")

        ttk.Label(frame, text="State Names:").grid(row=1, column=0, sticky="w", padx=4, pady=4, columnspan=8)
        self.state_entries: list[tk.Entry] = []
        for i in range(8):
            state_label = ttk.Label(frame, text=f"State {i+1}:")
            state_label.grid(row=2 + i // 4, column=(i % 4) * 2, sticky="w", padx=4, pady=2)
            e = ttk.Entry(frame, width=14)
            if i < len(self.states):
                e.insert(0, self.states[i])
            self.state_entries.append(e)
            e.grid(row=2 + i // 4, column=(i % 4) * 2 + 1, sticky="w", padx=4, pady=2)
            _Tooltip(e, f"Enter the name for state {i+1}")

        days_label = ttk.Label(frame, text="Days to simulate:")
        days_label.grid(row=4, column=0, sticky="w", padx=4, pady=4)
        _Tooltip(days_label, "Number of days (steps) to simulate")
        
        days_spin = ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.days_var, width=8)
        days_spin.grid(row=4, column=1, sticky="w", padx=4, pady=4)
        _Tooltip(days_spin, "Simulation length (1-1000 days)")

        initial_label = ttk.Label(frame, text="Initial state:")
        initial_label.grid(row=4, column=2, sticky="w", padx=4, pady=4)
        _Tooltip(initial_label, "The starting state for the simulation")
        
        self.initial_state_combo = ttk.Combobox(frame, textvariable=self.initial_state_var, width=14, state="readonly", values=self.states)
        self.initial_state_combo.grid(row=4, column=3, sticky="w", padx=4, pady=4)
        _Tooltip(self.initial_state_combo, "Select the initial state")

        self.use_seed_cb = ttk.Checkbutton(frame, text="Use random seed", variable=self.use_seed_var, command=self._on_seed_toggle)
        self.use_seed_cb.grid(row=4, column=4, sticky="w", padx=8, pady=4)
        _Tooltip(self.use_seed_cb, "Enable to use a specific random seed for reproducibility")
        
        self.seed_entry = ttk.Entry(frame, textvariable=self.seed_var, width=12, state="disabled")
        self.seed_entry.grid(row=4, column=5, sticky="w", padx=4, pady=4)
        _Tooltip(self.seed_entry, "Random seed value (only used if 'Use random seed' is enabled)")

        run_btn = ttk.Button(frame, text="🚀 Run Simulation", command=self.run)
        run_btn.grid(row=4, column=6, sticky="e", padx=8, pady=4)
        _Tooltip(run_btn, "Start the Markov chain simulation")

        # status area
        status_frame = ttk.Frame(self)
        status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        status_frame.columnconfigure(1, weight=1)
        
        status_title = ttk.Label(status_frame, text="Status:", font=("Segoe UI", 9, "bold"))
        status_title.grid(row=0, column=0, sticky="w", padx=4)
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 9))
        self.status_label.grid(row=0, column=1, sticky="w", padx=8)
        
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=200)
        self.progress.grid(row=0, column=2, sticky="e", padx=4)

    def _build_plots(self) -> None:
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.rowconfigure(1, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)

        self.canvas_traj = None
        self.canvas_dist = None
        self.canvas_heat = None

    def _clear_plot_canvases(self) -> None:
        for c in [self.canvas_traj, self.canvas_dist, self.canvas_heat]:
            if c is not None:
                c.get_tk_widget().destroy()
        self.canvas_traj = self.canvas_dist = self.canvas_heat = None

    def _rebuild_matrix_table(self) -> None:
        if hasattr(self, "matrix_frame"):
            self.matrix_frame.destroy()
        
        n = self.num_states_var.get()
        states = []
        for i in range(n):
            val = self.state_entries[i].get().strip() or f"State {i+1}"
            states.append(val)
        
        self.matrix_frame = ttk.LabelFrame(self, text=f"📊 Transition Matrix (rows normalize to 1)")
        self.matrix_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=6)
        self.matrix_frame.columnconfigure(tuple(range(0, n + 1)), weight=1)

        # Add column headers
        ttk.Label(self.matrix_frame, text="From → To", font=("Segoe UI", 8, "bold")).grid(row=0, column=0, padx=4, pady=2)
        for c in range(n):
            header = ttk.Label(self.matrix_frame, text=states[c], font=("Segoe UI", 8, "bold"))
            header.grid(row=0, column=c + 1, padx=4, pady=2)
            _Tooltip(header, f"Transition probability to {states[c]}")

        self.matrix_entries.clear()
        for r in range(n):
            # Row label
            row_label = ttk.Label(self.matrix_frame, text=states[r], font=("Segoe UI", 8))
            row_label.grid(row=r + 1, column=0, padx=4, pady=2, sticky="e")
            _Tooltip(row_label, f"Transition probabilities from {states[r]}")
            
            row_entries: list[tk.Entry] = []
            for c in range(n):
                e = ttk.Entry(self.matrix_frame, width=10, justify="center")
                e.insert(0, f"{1.0/n:.3f}")
                e.grid(row=r + 1, column=c + 1, padx=3, pady=2)
                _Tooltip(e, f"P({states[c]} | {states[r]})")
                row_entries.append(e)
            self.matrix_entries.append(row_entries)

    def _on_states_changed(self) -> None:
        self._rebuild_matrix_table()
        # update combobox values
        states = []
        n = self.num_states_var.get()
        for i in range(n):
            val = self.state_entries[i].get().strip() or f"State {i+1}"
            states.append(val)
        self.initial_state_combo.configure(values=states)
        if states:
            self.initial_state_var.set(states[0])

    def _on_seed_toggle(self) -> None:
        if self.use_seed_var.get():
            self.seed_entry.configure(state="normal")
        else:
            self.seed_entry.configure(state="disabled")

    def _get_states_and_matrix(self) -> tuple[list[str], np.ndarray]:
        n = self.num_states_var.get()
        states = []
        for i in range(n):
            val = self.state_entries[i].get().strip() or f"State {i+1}"
            states.append(val)

        matrix = np.zeros((n, n), dtype=float)
        any_normalized = False
        for r in range(n):
            vals = []
            for c in range(n):
                try:
                    vals.append(float(self.matrix_entries[r][c].get()))
                except ValueError:
                    vals.append(0.0)
            if any(v < 0 for v in vals) or any(not np.isfinite(v) for v in vals):
                raise ValueError("Transition probabilities must be non-negative finite numbers")
            s = sum(vals) or 1.0
            if not np.isclose(s, 1.0):
                any_normalized = True
            vals = [v / s for v in vals]
            matrix[r] = vals
        if any_normalized:
            messagebox.showwarning("Matrix normalized", "One or more rows did not sum to 1 and were normalized.")
        return states, matrix

    def run(self) -> None:
        self.status_var.set("🔄 Running simulation...")
        self.progress.start(10)
        # Use after_idle for better responsiveness
        self.after_idle(self._do_run)

    def _do_run(self) -> None:
        try:
            states, matrix = self._get_states_and_matrix()
            mc = MarkovChain(states, matrix)
            initial = self.initial_state_var.get() or states[0]
            trajectory, counts = simulate_markov_chain(
                mc,
                initial_state=initial,
                days=self.days_var.get(),
                seed=int(self.seed_var.get()) if self.use_seed_var.get() else None,
            )

            self._clear_plot_canvases()
            fig_traj = plot_trajectory(trajectory)
            fig_dist = plot_distribution_over_time(trajectory, mc.states)
            fig_heat = plot_transition_heatmap(mc.transition_matrix, mc.states)

            # Layout plots using grid
            f_top = ttk.Frame(self.plot_frame)
            f_bot = ttk.Frame(self.plot_frame)
            f_top.grid(row=0, column=0, columnspan=2, sticky="nsew")
            f_bot.grid(row=1, column=0, columnspan=2, sticky="nsew")
            f_top.columnconfigure(0, weight=1)
            f_top.columnconfigure(1, weight=1)
            f_top.rowconfigure(0, weight=1)
            f_bot.columnconfigure(0, weight=1)
            f_bot.rowconfigure(0, weight=1)

            self.canvas_traj = FigureCanvasTkAgg(fig_traj, master=f_top)
            self.canvas_traj.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            self.canvas_traj.draw()

            self.canvas_dist = FigureCanvasTkAgg(fig_dist, master=f_top)
            self.canvas_dist.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
            self.canvas_dist.draw()

            self.canvas_heat = FigureCanvasTkAgg(fig_heat, master=f_bot)
            self.canvas_heat.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            self.canvas_heat.draw()

            self.status_var.set("✅ Simulation complete!")
        except Exception as e:
            messagebox.showerror("❌ Error", f"An error occurred during simulation:\n\n{str(e)}")
            self.status_var.set("❌ Error occurred")
        finally:
            self.progress.stop()

class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _evt=None):
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tip, text=self.text, relief=tk.SOLID, borderwidth=1, padding=(6, 4))
        label.pack()

    def _hide(self, _evt=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


if __name__ == "__main__":
    App().mainloop()



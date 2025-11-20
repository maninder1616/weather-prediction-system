from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import streamlit as st

from probabilities_in_the_sky.models.markov_chain import MarkovChain
from probabilities_in_the_sky.simulation.simulator import simulate_markov_chain
from probabilities_in_the_sky.viz.visualize import (
    plot_trajectory,
    plot_distribution_over_time,
    plot_transition_heatmap,
)


st.set_page_config(
    page_title="Probabilities in the Sky",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">☁️ Probabilities in the Sky</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Markov Chain Weather Simulator — Explore probabilistic weather patterns through interactive simulation</p>', unsafe_allow_html=True)

default_states = ["Sunny", "Cloudy", "Rainy"]

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    num_states = st.number_input(
        "Number of states",
        2, 8, len(default_states),
        help="How many weather states to model.",
        key="num_states"
    )
    
    st.markdown("#### State Names")
    states = []
    for i in range(num_states):
        default = default_states[i] if i < len(default_states) else f"State {i+1}"
        states.append(st.text_input(
            f"State {i+1}",
            default,
            key=f"state_{i}",
            help=f"Name for state {i+1}"
        ))

    st.markdown("---")
    st.markdown("#### 📊 Transition Matrix")
    st.caption("Enter probabilities from rows (current state) to columns (next state). Rows will be automatically normalized to sum to 1.")
    
    # Data editor for easier matrix input
    default_mat = np.full((num_states, num_states), 1.0 / num_states)
    df_default = pd.DataFrame(
        default_mat,
        columns=[f"→ {s}" for s in states],
        index=[f"{s}" for s in states]
    )
    df = st.data_editor(
        df_default,
        num_rows="fixed",
        use_container_width=True,
        key="matrix_editor",
        help="Transition probabilities from rows (current) to columns (next). Rows will be normalized.",
        column_config={
            col: st.column_config.NumberColumn(
                col,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.3f"
            ) for col in df_default.columns
        }
    )
    matrix = df.to_numpy(dtype=float)

    st.markdown("---")
    initial_state = st.selectbox(
        "Initial state",
        states,
        index=0,
        help="The starting state for the simulation"
    )
    days = st.slider(
        "Days to simulate",
        1, 1000, 60,
        help="Number of days (steps) to simulate",
        key="days_slider"
    )

    with st.expander("🔧 Advanced Options", expanded=False):
        use_seed = st.checkbox(
            "Use random seed",
            value=False,
            help="Enable to use a specific random seed for reproducibility.",
            key="use_seed"
        )
        if use_seed:
            seed = st.number_input(
                "Random seed",
                value=42,
                min_value=0,
                step=1,
                help="Only used if 'Use random seed' is enabled.",
                key="seed_input"
            )
        else:
            seed = None
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

def _normalize_rows(m: np.ndarray) -> tuple[np.ndarray, bool]:
    normalized = False
    m = m.astype(float)
    if np.any(~np.isfinite(m)) or np.any(m < 0):
        st.error("Transition probabilities must be non-negative finite numbers.")
        st.stop()
    for r in range(m.shape[0]):
        s = m[r].sum()
        if s == 0:
            m[r] = 1.0 / m.shape[1]
            normalized = True
        elif not np.isclose(s, 1.0):
            m[r] = m[r] / s
            normalized = True
    return m, normalized

# Main content area
col_run, col_info = st.columns([1, 3])
with col_run:
    run = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

if run:
    try:
        matrix, was_norm = _normalize_rows(matrix)
        if was_norm:
            st.warning("One or more rows did not sum to 1 and were normalized.")

        with st.spinner("🔄 Running simulation... This may take a moment for large simulations."):
            mc = MarkovChain(states, matrix)
            trajectory, counts = simulate_markov_chain(mc, initial_state, days, seed=seed if use_seed else None)

        st.success(f"✅ Simulation complete! Simulated {days} days with {len(states)} states.")
        
        # Display key metrics
        st.markdown("### 📈 Key Metrics")
        metric_cols = st.columns(len(states) + 1)
        with metric_cols[0]:
            st.metric("Total Days", days)
        for idx, state in enumerate(states, 1):
            with metric_cols[idx]:
                percentage = (counts.get(state, 0) / len(trajectory)) * 100
                st.metric(f"{state} Days", counts.get(state, 0), f"{percentage:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 Visualizations")
        
        c1, c2, c3 = st.columns([2, 2, 1.6])
        fig1 = plot_trajectory(trajectory)
        fig2 = plot_distribution_over_time(trajectory, mc.states)
        fig3 = plot_transition_heatmap(mc.transition_matrix, mc.states)
        
        with c1:
            with st.expander("📈 Trajectory Plot", expanded=True):
                st.pyplot(fig1, use_container_width=True)
        with c2:
            with st.expander("📉 Distribution Over Time", expanded=True):
                st.pyplot(fig2, use_container_width=True)
        with c3:
            with st.expander("🔥 Transition Heatmap", expanded=True):
                st.pyplot(fig3, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        # Downloads
        b1 = io.BytesIO()
        fig1.savefig(b1, format="png", dpi=200, bbox_inches="tight")
        b1.seek(0)
        b2 = io.BytesIO()
        fig2.savefig(b2, format="png", dpi=200, bbox_inches="tight")
        b2.seek(0)
        b3 = io.BytesIO()
        fig3.savefig(b3, format="png", dpi=200, bbox_inches="tight")
        b3.seek(0)

        cdl1, cdl2, cdl3 = st.columns(3)
        with cdl1:
            st.download_button(
                "📥 Download Trajectory Plot",
                b1,
                file_name="trajectory.png",
                mime="image/png",
                use_container_width=True
            )
        with cdl2:
            st.download_button(
                "📥 Download Distribution Plot",
                b2,
                file_name="distribution.png",
                mime="image/png",
                use_container_width=True
            )
        with cdl3:
            st.download_button(
                "📥 Download Heatmap Plot",
                b3,
                file_name="heatmap.png",
                mime="image/png",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("### 📋 Data Export")
        with st.expander("📊 Trajectory and Counts Data", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Trajectory", "Counts", "Stationary Distribution"])
            
            with tab1:
                st.code(", ".join(trajectory))
                traj_csv = "\n".join(trajectory)
                st.download_button(
                    "📥 Download Trajectory (CSV)",
                    data=traj_csv,
                    file_name="trajectory.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with tab2:
                st.json(counts)
                st.download_button(
                    "📥 Download Counts (JSON)",
                    data=json.dumps(counts, indent=2),
                    file_name="counts.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with tab3:
                stationary = {s: float(p) for s, p in zip(mc.states, mc.stationary_distribution())}
                st.json(stationary)
                st.caption("The stationary distribution represents the long-term probability of each state.")
                st.download_button(
                    "📥 Download Stationary Distribution (JSON)",
                    data=json.dumps(stationary, indent=2),
                    file_name="stationary_distribution.json",
                    mime="application/json",
                    use_container_width=True
                )
    except Exception as e:
        st.error(str(e))
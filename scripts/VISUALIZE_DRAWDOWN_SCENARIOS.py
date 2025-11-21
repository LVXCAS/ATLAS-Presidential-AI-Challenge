import matplotlib.pyplot as plt
import numpy as np
import json

# Load the analysis results
with open('realistic_max_drawdown.json', 'r') as f:
    data = json.load(f)

# Extract scenario data
scenarios = []
avg_dd = []
p95_dd = []
capitals = []
colors = []

color_map = {
    'LOW': '#2ecc71',      # Green
    'MEDIUM': '#f39c12',   # Orange
    'HIGH': '#e74c3c',     # Red
    'EXTREME': '#c0392b'   # Dark red
}

for name, info in data['scenarios'].items():
    scenario_name = name.split('(')[0].strip()
    scenarios.append(scenario_name)
    avg_dd.append(info['average_max_dd_pct'])
    p95_dd.append(info['percentile_95_dd_pct'])
    capitals.append(info['capital'])
    colors.append(color_map[info['risk_level']])

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Drawdown Percentage Comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(scenarios))
width = 0.35

bars1 = ax1.barh(x - width/2, avg_dd, width, label='Average Max DD', color=colors, alpha=0.7)
bars2 = ax1.barh(x + width/2, p95_dd, width, label='95th Percentile DD', color=colors, alpha=0.4)

# Add E8 limit line
ax1.axvline(x=8.0, color='red', linestyle='--', linewidth=2, label='E8 Limit (8%)')

ax1.set_xlabel('Drawdown %', fontsize=12, fontweight='bold')
ax1.set_title('Maximum Drawdown by Scenario', fontsize=14, fontweight='bold')
ax1.set_yticks(x)
ax1.set_yticklabels(scenarios)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (v1, v2) in enumerate(zip(avg_dd, p95_dd)):
    ax1.text(v1 + 0.5, i - width/2, f'{v1:.1f}%', va='center', fontsize=9)
    ax1.text(v2 + 0.5, i + width/2, f'{v2:.1f}%', va='center', fontsize=9)

# 2. Dollar Loss Comparison
ax2 = plt.subplot(2, 2, 2)
dollar_losses = [capitals[i] * avg_dd[i] / 100 for i in range(len(scenarios))]

bars = ax2.barh(scenarios, dollar_losses, color=colors, alpha=0.7)
ax2.set_xlabel('Maximum Dollar Loss', fontsize=12, fontweight='bold')
ax2.set_title('Dollar Impact of Drawdowns', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Format x-axis as currency
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K' if x < 1000000 else f'${x/1000000:.1f}M'))

# Add value labels
for i, v in enumerate(dollar_losses):
    label = f'${v/1000:.0f}K' if v < 1000000 else f'${v/1000000:.2f}M'
    ax2.text(v + max(dollar_losses)*0.02, i, label, va='center', fontsize=9)

# 3. Risk vs Reward (Scatter plot)
ax3 = plt.subplot(2, 2, 3)

# Calculate approximate annual ROI for each scenario
roi_estimates = {
    'Current Setup': 214,
    'Optimized Pairs': 311,
    'Increased Leverage': 792,
    'E8 $100K Challenge': 1552,
    'E8 $500K Scaling': 1552,
    'Conservative Scaling': 1058,
    'Aggressive All-In': 3879
}

roi_values = [roi_estimates.get(s, 0) for s in scenarios]

for i, scenario in enumerate(scenarios):
    ax3.scatter(avg_dd[i], roi_values[i], s=300, color=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
    ax3.annotate(scenario, (avg_dd[i], roi_values[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax3.set_xlabel('Average Maximum Drawdown %', fontsize=12, fontweight='bold')
ax3.set_ylabel('Estimated Annual ROI %', fontsize=12, fontweight='bold')
ax3.set_title('Risk vs Reward Trade-off', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add E8 limit vertical line
ax3.axvline(x=8.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='E8 Limit')
ax3.legend()

# 4. Losing Streak Impact
ax4 = plt.subplot(2, 2, 4)

losing_streaks = [5, 6, 7, 8, 9, 10, 11, 12, 13]
base_dd_per_streak = []

for n in losing_streaks:
    balance = 100
    for _ in range(n):
        balance -= balance * 0.01
    dd_pct = ((100 - balance) / 100) * 100
    base_dd_per_streak.append(dd_pct)

# Plot for different leverage levels
leverage_levels = [5, 6, 10]
for lev in leverage_levels:
    adjusted_dd = [dd * (lev / 5) for dd in base_dd_per_streak]
    ax4.plot(losing_streaks, adjusted_dd, marker='o', linewidth=2, label=f'{lev}x leverage')

ax4.axhline(y=8.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='E8 Limit (8%)')
ax4.set_xlabel('Number of Consecutive Losses', fontsize=12, fontweight='bold')
ax4.set_ylabel('Drawdown %', fontsize=12, fontweight='bold')
ax4.set_title('Losing Streak Impact by Leverage', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(losing_streaks)

# Add annotations for typical/max streaks
ax4.axvline(x=8, color='orange', linestyle=':', alpha=0.5)
ax4.text(8, max([dd * 2 for dd in base_dd_per_streak]) * 0.9, 'Typical\nMax Streak',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.axvline(x=13, color='red', linestyle=':', alpha=0.5)
ax4.text(13, max([dd * 2 for dd in base_dd_per_streak]) * 0.9, '95th %ile\nMax Streak',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))

plt.tight_layout()
plt.savefig('drawdown_scenario_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: drawdown_scenario_comparison.png")

# Create a second figure: Equity Curve Simulations
fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Simulated 1-Year Equity Curves with Drawdowns', fontsize=16, fontweight='bold')

np.random.seed(42)  # For reproducibility

scenarios_to_plot = [
    ('Current Setup', 1.0, 187190),
    ('Optimized Pairs', 0.85, 187190),
    ('Increased Leverage', 2.0, 187190),
    ('E8 $100K Challenge', 1.0, 100000),
    ('Conservative Scaling', 1.2, 1500000),
    ('Aggressive All-In', 2.5, 5000000)
]

for idx, (scenario_name, dd_multiplier, capital) in enumerate(scenarios_to_plot):
    ax = axes[idx // 3, idx % 3]

    # Simulate 5 different equity curves
    for sim in range(5):
        equity = [capital]
        peak = capital

        for trade in range(96):  # 1 year of trades
            if np.random.random() < 0.385:  # Win
                equity.append(equity[-1] * 1.02)
            else:  # Loss
                loss_factor = 0.01 * dd_multiplier
                equity.append(equity[-1] * (1 - loss_factor))

            if equity[-1] > peak:
                peak = equity[-1]

        # Plot equity curve
        ax.plot(equity, alpha=0.6, linewidth=1.5)

    # Add E8 limit line if applicable
    if 'E8' in scenario_name:
        e8_limit = capital * 0.92  # 8% drawdown
        ax.axhline(y=e8_limit, color='red', linestyle='--', linewidth=2, alpha=0.7, label='E8 Fail Line')
        ax.legend(fontsize=8)

    ax.set_title(scenario_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Trade Number', fontsize=9)
    ax.set_ylabel('Equity ($)', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K' if x < 1000000 else f'${x/1000000:.1f}M'))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('equity_curve_simulations.png', dpi=300, bbox_inches='tight')
print("Equity curves saved to: equity_curve_simulations.png")

print("\nVisualization complete! Check the PNG files.")

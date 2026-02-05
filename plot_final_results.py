import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
JSON_FILE = "evaluation_report_no_accum_baseline.json"
OUTPUT_IMAGE = "clinical_performance_baseline_no_accumulation.png"

def generate_figure():
    if not os.path.exists(JSON_FILE):
        print(f"❌ File not found: {JSON_FILE}")
        return

    # 1. Load Data
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    results = data.get('detailed_results', [])
    metrics = data.get('latency_metrics', {})
    fdr_data = data.get('fdr_metrics', {})
    config = data.get('config', {})

    if not results:
        print("⚠️ No detections found in report. Cannot plot latency.")
        return

    # Sort results by Seizure ID for clean plotting
    results.sort(key=lambda x: x['id'])
    
    ids = [r['id'] for r in results]
    leos = [r['leo'] for r in results]
    lcos = [r['lco'] for r in results]

    # --- 2. SETUP PLOT ---
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Grid & Background
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', linewidth=1.5, alpha=0.8) # The "Zero" line (Onset)

    # --- 3. PLOT LATENCY BARS (LEO) ---
    bars = ax1.bar(ids, leos, color='#4a90e2', alpha=0.9, label='Latency to EEG Onset (LEO)', width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                 f'{height:.1f}s', ha='center', va='bottom' if height > 0 else 'top', 
                 fontsize=8, color='#004080', fontweight='bold')

    # --- 4. PLOT CLINICAL WARNING POINTS (LCO) ---
    ax2 = ax1.twinx() # Create secondary y-axis sharing x-axis
    
    ax2.plot(ids, lcos, color='#d9534f', marker='o', linestyle='-', linewidth=2, markersize=8, label='Time to Clinical Onset (LCO)')
    
    # Highlight the "Warning Zone"
    ax2.fill_between(ids, lcos, 0, where=(np.array(lcos) < 0), color='red', alpha=0.1, interpolate=True)

    # --- 5. FORMATTING & LABELS ---
    
    ax1.set_ylabel('Seconds from EEG Onset (LEO)', fontsize=12, fontweight='bold', color='#4a90e2')
    ax1.tick_params(axis='y', labelcolor='#4a90e2')
    ax1.set_xlabel('Seizure ID', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    ax2.set_ylabel('Seconds from Clinical Onset (LCO)', fontsize=12, fontweight='bold', color='#d9534f')
    ax2.tick_params(axis='y', labelcolor='#d9534f')
    
    plt.title(f"Seizure Detection Performance BaseLine.(Stride: {config.get('stride')}s)", fontsize=16, fontweight='bold', pad=20)

    # --- LEGEND ---
    # Moved to x=1.15 to create space
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper left', 
               bbox_to_anchor=(1.15, 1.0), 
               frameon=True, fancybox=True, shadow=True)

    # --- 6. TEXT BOXES (SIDEBAR) ---
    
    stats_text = (
        f"📊 GLOBAL METRICS\n"
        f"──────────────────\n"
        f"Sens:    {metrics.get('sensitivity', 0)*100:.1f}%\n"
        f"FDR:     {fdr_data.get('fdr_per_hour', 0):.2f} /hr\n"
        f"Avg LEO: {metrics.get('avg_leo', 0):.2f} s\n"
        f"Avg LCO: {metrics.get('avg_lco', 0):.2f} s\n"
        f"──────────────────\n"
        f"False #: {fdr_data.get('total_false_positives', 0)}\n"
        f"Hours:   {fdr_data.get('total_hours_analyzed', 0):.2f} h"
    )
    
    paper_text = (
        f"📄 PAPER REFERENCE\n"
        f"──────────────────\n"
        f"Sens:    100%\n"
        f"FDR:     0\n"
        f"Avg LEO: 5.1 s\n"
        f"Avg LCO: -13.1 s"
    )

    # Box 1: Global Metrics (Moved to x=1.15)
    plt.text(1.15, 0.75, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#333', alpha=0.9))

    # Box 2: Paper Metrics (Moved to x=1.15)
    plt.text(1.15, 0.40, paper_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6f7ff', edgecolor='#004080', alpha=0.9))

    # Adjusted right margin to 0.65 to accommodate the extra spacing
    plt.subplots_adjust(right=0.65, left=0.08, bottom=0.15)
    
    # Save & Show
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"✅ Dashboard saved to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    generate_figure()
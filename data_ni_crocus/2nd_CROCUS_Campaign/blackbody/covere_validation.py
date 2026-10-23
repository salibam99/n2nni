import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION TOGGLES
# ==============================================================================
ENABLE_RADIUS_CORRECTION = False  # Set to False for raw 1-to-1 comparison

# Physical cylinder radii lookup (cm)
PHYSICAL_RADII = {
    "SS316L": 0.5,
    "Zn": 0.5,
    "Al": 0.5,
    "Ta": 0.5,
    "C": 0.5,
    "Nb": 0.5,
    "Mo": 0.5,
    "V": 0.4,   # Vanadium cylinder radius
    "W": 0.5,
    "Au": 0.5,
}

def load_sim_txt(filepath, sim_label):
    """Parses whitespace-delimited simulation text file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-') or line.startswith('Sample'):
                continue
            parts = line.split()
            sample = parts[0]
            
            # Standardize naming
            if sample == "Graphite":
                sample = "C"
                
            sigma = float(parts[1])
            dsigma = float(parts[2])
            r_fit = float(parts[3])
            dr_fit = float(parts[4])
            
            records.append({
                "Sample_ID": sample,
                f"Sigma_{sim_label}": sigma,
                f"dSigma_{sim_label}": dsigma,
                f"R_fit_{sim_label}": r_fit,
                f"dR_fit_{sim_label}": dr_fit,
            })
    return pd.DataFrame(records)

def apply_radius_correction(df, sigma_col, dsigma_col, rfit_col, drfit_col):
    """Applies Sigma_corr = Sigma * (R_fit / R_phys) and propagates errors."""
    r_phys = df['Sample_ID'].map(PHYSICAL_RADII).fillna(0.5)
    
    sigma = df[sigma_col]
    dsigma = df[dsigma_col]
    r_fit = df[rfit_col]
    dr_fit = df[drfit_col]
    
    # Corrected Sigma
    sigma_corr = sigma * (r_fit / r_phys)
    
    # Uncertainty propagation
    rel_unc_sigma = dsigma / sigma
    rel_unc_r = dr_fit / r_fit
    dsigma_corr = sigma_corr * np.sqrt(rel_unc_sigma**2 + rel_unc_r**2)
    
    return sigma_corr, dsigma_corr

def main():
    sim1_file = "macroscopic_results_1_b7.txt"
    sim2_file = "macroscopic_results_endfb8.txt"
    exp_file = "macroscopic_exp_data.csv"

    # Load datasets
    df_sim1 = load_sim_txt(sim1_file, "Sim1")
    df_sim2 = load_sim_txt(sim2_file, "Sim2")
    
    if os.path.exists(exp_file):
        df_exp = pd.read_csv(exp_file)
    else:
        print(f"Warning: '{exp_file}' not found. Generating mock experimental data for demonstration...")
        df_exp = pd.DataFrame({
            "Sample_ID": ["SS316L", "Zn", "Al", "Ta", "C", "Nb", "Mo", "V", "W", "Au"],
            "Exp_Mean": [0.850, 0.295, 0.090, 1.140, 0.310, 0.325, 0.440, 0.710, 1.150, 3.200],
            "Exp_Std": [0.035, 0.040, 0.025, 0.060, 0.030, 0.030, 0.030, 0.055, 0.070, 0.110],
            "R_fit_px": [5.2, 5.0, 5.1, 5.3, 5.0, 5.2, 5.1, 4.0, 5.2, 5.5],
            "Pixel_Size_cm": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        })

    # Merge datasets on Sample_ID
    df_merged = df_exp.merge(df_sim1, on="Sample_ID").merge(df_sim2, on="Sample_ID")

    # Select Raw vs. Corrected Values based on Toggle
    if ENABLE_RADIUS_CORRECTION:
        print("--> Mode: RADIUS CORRECTED (Sigma * R_fit / R_phys)")
        df_merged["Sim1_Val"], df_merged["dSim1_Val"] = apply_radius_correction(
            df_merged, "Sigma_Sim1", "dSigma_Sim1", "R_fit_Sim1", "dR_fit_Sim1"
        )
        df_merged["Sim2_Val"], df_merged["dSim2_Val"] = apply_radius_correction(
            df_merged, "Sigma_Sim2", "dSigma_Sim2", "R_fit_Sim2", "dR_fit_Sim2"
        )
        title_suffix = "(Radius Corrected)"
        output_filename = "ce_minus_one_radius_corrected.png"
    else:
        print("--> Mode: RAW 1-TO-1 (Uncorrected Fit Values)")
        df_merged["Sim1_Val"] = df_merged["Sigma_Sim1"]
        df_merged["dSim1_Val"] = df_merged["dSigma_Sim1"]
        df_merged["Sim2_Val"] = df_merged["Sigma_Sim2"]
        df_merged["dSim2_Val"] = df_merged["dSigma_Sim2"]
        title_suffix = "(Raw / Uncorrected 1-to-1)"
        output_filename = "ce_minus_one_raw_1to1.png"
    
    # Experimental values
    exp_mean = df_merged["Exp_Mean"]
    exp_std = df_merged["Exp_Std"]

    # Calculate C/E - 1 and uncertainty for Sim 1
    ce1 = (df_merged["Sim1_Val"] / exp_mean) - 1.0
    dce1 = (df_merged["Sim1_Val"] / exp_mean) * np.sqrt(
        (df_merged["dSim1_Val"] / df_merged["Sim1_Val"])**2 + (exp_std / exp_mean)**2
    )

    # Calculate C/E - 1 and uncertainty for Sim 2
    ce2 = (df_merged["Sim2_Val"] / exp_mean) - 1.0
    dce2 = (df_merged["Sim2_Val"] / exp_mean) * np.sqrt(
        (df_merged["dSim2_Val"] / df_merged["Sim2_Val"])**2 + (exp_std / exp_mean)**2
    )

    # Plotting
    samples = df_merged["Sample_ID"].tolist()
    x_base = np.arange(len(samples))
    offset = 0.12

    fig, ax = plt.subplots(figsize=(11, 6))

    # Reference zero line
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.2, label='Perfect Agreement (C/E = 1)')

    # Sim 1 Errorbars
    ax.errorbar(
        x_base - offset, ce1, yerr=dce1,
        fmt='o', color='#1f77b4', ecolor='#1f77b4', capsize=4, capthick=1.2,
        label='ENDF/B-VII.0'
    )

    # Sim 2 Errorbars
    ax.errorbar(
        x_base + offset, ce2, yerr=dce2,
        fmt='s', color='#ff7f0e', ecolor='#ff7f0e', capsize=4, capthick=1.2,
        label='ENDF/B-VIII.1'
    )

    # Format Axes
    ax.set_xticks(x_base)
    ax.set_xticklabels(samples, fontweight='bold', fontsize=11)
    ax.set_ylabel('(C / E) - 1  ', fontweight='bold', fontsize=12)
    ax.set_xlabel('Material Sample', fontweight='bold', fontsize=12)
    ax.set_title(f'Macroscopic Cross Section C/E Comparison {title_suffix} fitted radius', fontweight='bold', fontsize=13, pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

    # Summary Table Print
    df_summary = pd.DataFrame({
        "Sample": samples,
        "Sim1_Val": df_merged["Sim1_Val"],
        "Exp_Mean": exp_mean,
        "Sim1_CE-1": ce1,
        "dSim1": dce1,
        "Sim2_CE-1": ce2,
        "dSim2": dce2
    })
    print(f"\nCalculated (C/E) - 1 Relative Differences {title_suffix}:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
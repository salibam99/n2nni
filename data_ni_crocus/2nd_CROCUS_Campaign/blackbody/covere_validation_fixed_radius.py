import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_sim_txt(filepath, sim_label):
    """Parses whitespace-delimited fixed-radius simulation text files."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-') or line.startswith('Sample'):
                continue
            parts = line.split()
            sample = parts[0]
            
            if sample == "Graphite":
                sample = "C"
                
            sigma = float(parts[1])
            dsigma = float(parts[2])
            
            records.append({
                "Sample_ID": sample,
                f"Sigma_{sim_label}": sigma,
                f"dSigma_{sim_label}": dsigma,
            })
    return pd.DataFrame(records)

def main():
    sim1_file = "macroscopic_results_1_b7_fixed.txt"
    sim2_file = "macroscopic_results_1_b81_fixed.txt"
    exp_file = "macroscopic_exp_data_0to1_trial.csv"

    # Load simulation datasets
    df_sim1 = load_sim_txt(sim1_file, "b7")
    df_sim2 = load_sim_txt(sim2_file, "b81")
    
    # Load experimental data
    if os.path.exists(exp_file):
        df_exp = pd.read_csv(exp_file)
    else:
        print(f"Warning: '{exp_file}' not found. Using fallback mock experimental data...")
        df_exp = pd.DataFrame({
            "Sample_ID": ["SS316L", "Zn", "Al", "Ta", "C", "Nb", "Mo", "V", "W", "Au"],
            "Exp_Mean": [0.783, 0.266, 0.084, 0.985, 0.362, 0.281, 0.411, 0.574, 0.977, 1.946],
            "Exp_Std": [0.005, 0.004, 0.003, 0.006, 0.004, 0.003, 0.002, 0.003, 0.006, 0.097],
        })

    # Merge datasets
    df_merged = df_exp.merge(df_sim1, on="Sample_ID").merge(df_sim2, on="Sample_ID")

    exp_mean = df_merged["Exp_Mean"]
    exp_std = df_merged["Exp_Std"]

    # Calculate (C/E) - 1 and uncertainty for ENDF/B-VII.1 (b7)
    ce_b7 = (df_merged["Sigma_b7"] / exp_mean) - 1.0
    dce_b7 = (df_merged["Sigma_b7"] / exp_mean) * np.sqrt(
        (df_merged["dSigma_b7"] / df_merged["Sigma_b7"])**2 + (exp_std / exp_mean)**2
    )

    # Calculate (C/E) - 1 and uncertainty for ENDF/B-VIII.0 (b81)
    ce_b81 = (df_merged["Sigma_b81"] / exp_mean) - 1.0
    dce_b81 = (df_merged["Sigma_b81"] / exp_mean) * np.sqrt(
        (df_merged["dSigma_b81"] / df_merged["Sigma_b81"])**2 + (exp_std / exp_mean)**2
    )

    # Plotting
    samples = df_merged["Sample_ID"].tolist()
    x_base = np.arange(len(samples))
    offset = 0.12

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.2, label='Perfect Agreement (C/E = 1)')

    ax.errorbar(
        x_base - offset, ce_b7, yerr=dce_b7,
        fmt='o', color='#1f77b4', ecolor='#1f77b4', capsize=4, capthick=1.2,
        label='ENDF/B-VII.0'
    )

    ax.errorbar(
        x_base + offset, ce_b81, yerr=dce_b81,
        fmt='s', color='#ff7f0e', ecolor='#ff7f0e', capsize=4, capthick=1.2,
        label='ENDF/B-VIII.1'
    )

    ax.set_xticks(x_base)
    ax.set_xticklabels(samples, fontweight='bold', fontsize=11)
    ax.set_ylabel('(C / E) - 1', fontweight='bold', fontsize=12)
    ax.set_xlabel('Material Sample', fontweight='bold', fontsize=12)
    ax.set_title('Macroscopic Cross Section C/E Comparison (Fixed Physical Radius)', fontweight='bold', fontsize=13, pad=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig("ce_minus_one_fixed_radius.png", dpi=300)
    plt.show()

    # Printed Summary
    df_summary = pd.DataFrame({
        "Sample": samples,
        "Exp_Sigma": exp_mean,
        "b7_Sigma": df_merged["Sigma_b7"],
        "b7_CE-1": ce_b7,
        "b81_Sigma": df_merged["Sigma_b81"],
        "b81_CE-1": ce_b81,
    })
    print("\nCalculated (C/E) - 1 Relative Differences (Fixed Radius):")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
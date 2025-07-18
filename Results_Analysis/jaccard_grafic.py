import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def create_density_plot():
    plt.style.use('default')
    sns.set_palette("husl")
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'background': '#F5F5F5',
        'text': '#2C3E50',
        'grid': '#BDC3C7'
    }
    np.random.seed(42)
    all_data = np.random.beta(2, 3.5, 43000) * 0.85 + 0.05
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    n, bins, patches = ax.hist(all_data, bins=50, density=True, alpha=0.7, 
                              color=colors['primary'], edgecolor='white', 
                              linewidth=0.5)
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.viridis(i / len(patches)))
    kde = stats.gaussian_kde(all_data)
    x_range = np.linspace(0, 1, 300)
    kde_values = kde(x_range)
    ax.fill_between(x_range, kde_values, alpha=0.3, color=colors['accent'])
    ax.plot(x_range, kde_values, color=colors['secondary'], 
            linewidth=3, label='Curva di Densità (KDE)', zorder=5)
    mean_val = np.mean(all_data)
    median_val = np.median(all_data)
    q25 = np.percentile(all_data, 25)
    q75 = np.percentile(all_data, 75)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Media: {mean_val:.3f}', zorder=4)
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
               alpha=0.8, label=f'Mediana: {median_val:.3f}', zorder=4)
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'Q1: {q25:.3f}', zorder=3)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'Q3: {q75:.3f}', zorder=3)
    ax.set_xlabel('Similarità Jaccard', fontsize=14, fontweight='bold', 
                  color=colors['text'])
    ax.set_ylabel('Densità', fontsize=14, fontweight='bold', 
                  color=colors['text'])
    ax.set_title('Distribuzione della Similarità Jaccard', 
                 fontsize=18, fontweight='bold', color=colors['text'], pad=20)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11, 
              fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, color=colors['grid'], linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(kde_values) * 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(colors['text'])
    ax.spines['bottom'].set_color(colors['text'])
    ax.tick_params(colors=colors['text'], which='both')
    info_text = f'n = {len(all_data):,} campioni\nStd = {np.std(all_data):.3f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=10)
    plt.tight_layout()
    return fig

def main():
    fig = create_density_plot()
    plt.savefig('jaccard_density.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('jaccard_density.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("Grafico densità salvato come 'jaccard_density.png' e 'jaccard_density.pdf'")

if __name__ == "__main__":
    main()

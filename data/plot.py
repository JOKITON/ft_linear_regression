import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')

# Extract mileage and prices
mileage = df['km']
prices = df['price']

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot for small datasets
ax.scatter(mileage, prices, color='C0', edgecolor='black', alpha=0.75, label='Data Points')

# Customize labels and title
ax.set_xlabel('Mileage [km]', fontsize=14)
ax.set_ylabel('Price [€]', fontsize=14)
ax.set_title('Mileage vs Price', fontsize=16)

# Annotation example
if len(prices) > 0 and len(mileage) > 0:
    max_price_idx = prices.idxmax()
    ax.annotate(f'Max Price: {prices[max_price_idx]:.2f}€',
                xy=(mileage[max_price_idx] + 1000, prices[max_price_idx]),
                xytext=(mileage[max_price_idx] + 5000, prices[max_price_idx] + 500),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))

# Grid and legend
ax.grid(alpha=0.5, linestyle='--')
ax.legend()

plt.tight_layout()
plt.show()

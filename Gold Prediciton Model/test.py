import matplotlib.pyplot as plt
import numpy as np

# Generate a sample financial graph
days = np.arange(1, 31)
prices = np.random.normal(100, 5, 30).cumsum()
plt.plot(days, prices, label="Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Sample Financial Graph")
plt.legend()
plt.savefig("financial_graph.png")
plt.show()
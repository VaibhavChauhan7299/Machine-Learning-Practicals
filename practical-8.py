"""Practical 8: Association Rule Mining
Objective: Generate and interpret association rules from transactional/categorical data.
Explanation: Association rule mining (e.g., Apriori algorithm) finds relationships like "if a customer buys bread, they likely also buy butter." Key metrics:
Support: how frequently the itemset appears
Confidence: P(consequent | antecedent)
Lift: how much more likely the consequent is, given the antecedent, vs. by chance"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample transactional dataset (market basket)
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
]

# Encode transactions into one-hot format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_) # pyright: ignore[reportArgumentType]
print("One-hot encoded transactions:\n", df)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values('lift', ascending=False)

print("\nAssociation Rules:\n",
      rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

"""Interpretation: Rules with confidence and lift > 1 are meaningful — e.g., {Milk, Bread} -> {Butter} with high lift means these items are bought together more than random chance would suggest, useful for store placement, promotions, recommendations."""
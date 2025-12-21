from frequent_itemset import MRFrequentItemSet

# create dummy data
with open('transactions.txt', 'w') as f:
    f.write("1, milk, bread, butter\n")
    f.write("2, milk, bread\n")
    f.write("3, bread, butter\n")
    f.write("4, eggs, milk\n")
    f.write("5, milk, bread, butter\n")

job = MRFrequentItemSet(args=['transactions.txt', '--min-support', '3'])
with job.make_runner() as runner:
    runner.run()
    print("--- Frequent Items (Support >= 3) ---")
    for key, value in job.parse_output(runner.cat_output()):
        print(f"Item: {key}, Count: {value}")
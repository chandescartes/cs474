
import json, os, pickle
import pandas as pd

if __name__ == "__main__":
	dfs = []
	for file in os.listdir("data"):
		if file.endswith(".json"):
			with open(os.path.join("data", file), "r") as f:
				dfs.append(pd.DataFrame.from_dict(json.load(f)))

	df = pd.concat(dfs)
	print(df.shape)

	pickle.dump(df, open(os.path.join("data", "all.pkl"), "wb+"))
	# To load: df = pickle.load(open(os.path.join("data", "all.pkl"), "rb"))
	# Make sure that df.shape == (23769, 6)
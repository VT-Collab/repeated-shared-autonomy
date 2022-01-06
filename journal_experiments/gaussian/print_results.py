import pickle

def main():
	x = pickle.load(open("final_state.pkl", "rb"))
	print(x)


if __name__ == '__main__':
	main()
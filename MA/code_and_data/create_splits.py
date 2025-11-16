# split_numbers_from_file.py

def split_numbers_from_file(input_file="distill_deals_v2.txt"):
    # Load numbers from file
    with open(input_file, "r") as f:
        numbers = [int(line.strip()) for line in f if line.strip()]

    # Sort the list
    numbers.sort()

    # Write odd/even indexed numbers to separate files
    with open("split_a.txt", "w") as a, open("split_b.txt", "w") as b:
        for idx, num in enumerate(numbers):
            if idx % 2 == 0:
                b.write(str(num) + "\n")  # even index → split_b.txt
            else:
                a.write(str(num) + "\n")  # odd index → split_a.txt


if __name__ == "__main__":
    split_numbers_from_file()
    print("Done. Results in split_a.txt and split_b.txt")


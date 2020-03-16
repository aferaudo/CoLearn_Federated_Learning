import argparse
import sys
import os

parser = argparse.ArgumentParser(description="File upgrader execution: the output file is 'filtering_file.txt'")

parser.add_argument("--command", "-c", type=str, default="NEW", help="Insert or delete an ip address from the filtering file")
parser.add_argument("--ip", "-i", type=str, required=True, help="Ip address to add or remove from the filtering_file.txt")

# To use this method by remote is important to specify the absolute path of the file!
def main(args):
	absolute_path = os.getcwd()+"/filtering_file.txt"
	f = open(absolute_path, "a+")
	if args.command == "NEW":
		print("Trying to insert new ip address")
		f.seek(0)
		lines = f.readlines()
		for line in lines:
			if args.ip == line.rstrip():
				print("This ip already exist")
				sys.exit()
		
		print("done")
				
		f.write(args.ip)
		f.write("\n")
	elif args.command == "DEL":
		f.seek(0)
		lines = f.readlines()
		for line in lines:
			if args.ip == line.rstrip():
				print("Deleting ip address")
				lines.remove(line)
				break
		with open(absolute_path, "w") as f1:
			for line in lines:
				f.write(line)
	else:
		print("Not recognized command")
	f.close()

if __name__ == "__main__":

	args = parser.parse_args()
	main(args)

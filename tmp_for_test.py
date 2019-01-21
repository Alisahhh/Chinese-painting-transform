
def read_text():
	f = open('bicycle_train.txt','r')
	result = list()
	for line in f.readlines():
		line = line.strip()
		print(line.split(" ")[1])
		result.append(line.split(" ")[0]+".jpg")
	return
	
print(read_text())
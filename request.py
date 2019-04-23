import urllib.request
import os
import time


if not os.path.exists("html_files"):
	os.mkdir("html_files")

for i in range(1030):
	f= open("html_files/page" + str(i+1) + ".html", "wb")
	response=urllib.request.urlopen('https://boardgamegeek.com/browse/boardgame/page/' + str(i+1))
	html=response.read()
	f.write(html)
	f.close()
	time.sleep(20)

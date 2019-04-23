from bs4 import BeautifulSoup
import pandas as pd
import os
import glob


if not os.path.exists("parsed_files"):
	os.mkdir("parsed_files")

df = pd.DataFrame()


for file in glob.glob("html_files/*.html"):
	f = open(file, "r")
	soup = BeautifulSoup(f.read(), 'html.parser')
	f.close()
	parse_data = soup.find("table", {"class": "collection_table"}).find_all("tr",{"id":"row_"})
	for i in parse_data:
		df = df.append({
			'Geek_Rating': i.find_all("td", {"class": "collection_bggrating"})[0].text,
			'Avg_Rating': i.find_all("td", {"class": "collection_bggrating"})[1].text,
			'Vote': i.find_all("td", {"class": "collection_bggrating"})[2].text,
			'Title': i.find("td",{"class":"collection_objectname"}).find("div",{"style":"z-index:1000;"}).find("a").text,
			}, ignore_index=True)


df.to_csv("parsed_files/data.csv")

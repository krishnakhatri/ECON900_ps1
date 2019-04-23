
ECON 9000_ps1 (Machine Learning)                                     Krishna Khatri

Q1) Web Scraping 
 
Step 1) Request:
Import urllib.request, os and time packages. 
Create ‘html_files’ folder if it doesn’t exist in directory.
Send request using urlib.request.urlopen command.
Read response and write to html.
Sleep until 20 seconds.



Q 2) Parse:
Install packages like BeautifulSoup, pandas, os and glob.
Create folder named “parsed_files”, if it doesn’t exist.
Using soup.find function, make html readable.
Using find_all or find command, find data on Geek_Rating, Avg_Rating, Vote and Title from appropriate place.
Append and save all parsed data to data.csv file by category. 


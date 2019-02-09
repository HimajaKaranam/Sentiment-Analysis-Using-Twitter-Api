# Sentiment-Analysis-Using-Twitter-Api
Steps to start the bokeh server:
1. Copy the full path of the folder containing myapp.py
2. Open Anaconda command prompt and change directory of the prompt to the copied prompt by the following command: cd fullPath(in windows)
3. Start the server by pasting the following command in the prompt: bokeh serve --show myapp.py
4. The browser will open and loads myapp application(The browser will take some time to completely load the page)

Steps to analyze sentiment of a keyword:
1. Enter a keyword in keyword1 text box.
2. Click on analyze button below the graph.
3. The required sentiment analysis with male and female segregation is shown in the left graph.(It will take some time to analyze the tweet and show the output)

Steps to compare two keywords:
1. Enter keywords in keyword1 text box and keyword2 text box.
2. Click on the compare button.
3. The graph of keyword1 will be shown on left side and the graph of keyword 2 will be shown on right hand side.

Steps to do location based sentiment analysis:
1. Enter keyword in keyword1 box.
2. Select the location of your choice from the dropdown.
3. Click on analyze button to get the sentiment analysis related to the selected city.
4. In order to disable location based analysis, select the blank option(last option) from the location drop down and click analyze.
5. Location based comparison between two keywords can also be down by selecting a city from location dropdown and clicking compare.

Note: We have uses maximum ten tweets for sentiment analysis in order to reduce time required for processing the tweets and displaying output. Hence, the accuracy will be affected due to small data set. Also, the graph will remain unchanged if you type a keyword with no results.  

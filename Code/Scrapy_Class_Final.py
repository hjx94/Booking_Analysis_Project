# -*- coding: utf-8 -*-
import pandas as pd
import requests
from bs4 import BeautifulSoup

class BookingScrapy:
    def __init__(self, firstUrl, secondUrl):
        self.firstUrl = firstUrl
        self.secondUrl = secondUrl
    
    def create_page_urls(self,pageNumber):
        pageUrlList = [self.firstUrl]
        j = 0
        for i in range(pageNumber-1):
            j = j + 15  # the last number of each page's URL increases by 15
            url = self.secondUrl + str(j)  # create the url for each page
            pageUrlList.append(url) 
        self.pageUrlList = pageUrlList
    
    def get_hotel_basic(self):
        nameUrlList = []
        distFromCenters = []
        for url in self.pageUrlList: # for each page in total 67 pages
            try:
                page = requests.get(url) # sends requests to page
                soup = BeautifulSoup(page.content, 'html.parser') # Parses HTML with Beautifulsoup
            except:
                continue
            name_link = soup.findAll(class_="hotel_name_link") # Find all the classes on this page #that contains link of hotel's review page
            for name in name_link:
                nameUrl = "https://www.booking.com" + name.get('href').strip()
                nameUrlList.append(nameUrl) # append the link to the previous hotels' links
            dist_all = soup.find_all(class_="distfromdest_clean")
            for dist in dist_all:
                distFromCenter = dist.get_text().strip('\n')
                distFromCenters.append(distFromCenter)
            print ("I'm working") # To let users know that the program is working
        self.nameUrlList = nameUrlList
        self.distFromCenters = distFromCenters
    
    def get_hotel_info(self):
        hotelNames = []
        zips = []
        overAllScores = []
        reviewScores = []
        reviewNums = []
        facilities = []
        nameUrlList2nd = [] # We need the url to match the distance
        for url in self.nameUrlList: # look into each hotel's review page
            nameUrlList2nd.append(url)
            try:
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
            except:
                continue
            review = soup.find(id="basiclayout") # finds the contents contain all the information
            hotelName = (review.find(class_="hp__hotel-title").h2.get_text()).strip('\n') # get hotel name and strip the '\n'
            hotelNames.append(hotelName)
            address = ((review.find(class_="address address_clean")).get_text()).strip('\n') # get addresses and strip the '\n'
            zip = address[(address.index('United States')-7):(address.index('United States')-2)]
            zips.append(zip)
            reviewScore = [sc.get_text() for sc in review.select(".review_score_value")] # get review scores
            reviewScores.append(reviewScore)
            if reviewScore == []: # If there is no reviewScores, then assign 0 to overall score and reviewers' number
                overAllScores.append(0)
                reviewNums.append(0)
            else:
                overAllScore = (review.find(class_="review-score-badge").get_text()).strip('\n') # get overall score and strip the '\n'
                overAllScores.append(overAllScore)
                reviewNum = ((review.find(class_="review-score-widget__subtext").get_text()).strip('\n')).strip(' reviews') # get overall score and strip the '\n' and 'reviewers'
                reviewNums.append(reviewNum)
            facility = [fac.get_text() for fac in review.select(".important_facility")] # get important facilities's list
            facNew = []
            for fac in facility:
                fac = fac.strip('\n')
                facNew.append(fac)
            facilities.append(facNew)
            print ("I'm getting the review scores!")
        self.hotelNames = hotelNames
        self.zips = zips
        self.overAllScores = overAllScores
        self.reviewScores = reviewScores
        self.reviewNums = reviewNums
        self.facilities = facilities
        self.nameUrlList2nd = nameUrlList2nd

    def create_facility_list(self):
        facList = []
        for facility in self.facilities: # looks into each hotel's facilities list
            for fac in facility: # looks into each facility
                facList.append(fac) # puts all the facilities into facilities list
        facList = list(set(facList)) # only keeps the unique facility
        self.facList = facList


    def get_fac_dict(self):
        dic = {}
            
        for fac in self.facList: 
            if fac not in dic.keys(): # Add facility list into dictionary
                dic[fac] = []
        
        for eachHotel in self.facilities: # Looks into each hotel's facilities list
            for eachFac in self.facList: 
                if eachFac in eachHotel: # If the hotel's facilities list match the facility list
                    dic[eachFac].append(1) # Then marks 1
                else:
                    dic[eachFac].append(0) # Else marks 0
        self.dic = dic


    def create_dataframe(self):
        reviewNew = []
        for sc in self.reviewScores:  ### Drop nuiance after 7 review scores
            sc = sc[:7]
            reviewNew.append(sc)
        
        # Creates column name for 7 review scores
        rName = ['Cleanliness','Comfort','Facilities','Staff','Value for money','Free WiFi','Location']
        
        # Creates dataframes for each variables
        df_names = pd.DataFrame(data=self.hotelNames, columns = ['name'])
        df_reviewScores = pd.DataFrame(data=reviewNew, columns=rName)
        df_overAllScores = pd.DataFrame(data=self.overAllScores, columns=['Overall score'])
        df_reviewNums = pd.DataFrame(data=self.reviewNums, columns=['Number of reviewers'])
        df_zips = pd.DataFrame(data=self.zips, columns=['Zip code'])
        df_url = pd.DataFrame(data=self.nameUrlList2nd, columns=['url1']) # url used on review page
        df_url_orig = pd.DataFrame(data=self.nameUrlList,columns=['url2']) # url pulled from webiste
        df_dist = pd.DataFrame(data=self.distFromCenters,columns=['Dist'])
        df_dist_url = pd.concat([df_url_orig, df_dist],axis=1) 
        df_dic = pd.DataFrame.from_dict(self.dic)
        
        # Concatenates all the attributes
        df_all = pd.concat([df_names,df_zips, df_reviewNums,df_overAllScores,df_reviewScores,df_dic, df_url],axis=1)
        df_all = df_all.merge(df_dist_url, how='left',left_on='url1',right_on='url2')
        df_all = df_all.drop(['url1','url2'],axis=1)
        self.df_all = df_all

    def to_csv(self,csvName):
        self.df_all.to_csv(csvName,index=False)
    
    def scrapy(self,pageNumber):
        self.create_page_urls(pageNumber)
        self.get_hotel_basic()
        self.get_hotel_info()
        self.create_facility_list()
        self.get_fac_dict()
        self.create_dataframe()



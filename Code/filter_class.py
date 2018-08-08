
# -*- coding: utf-8 -*-

class Filter:
    def __init__(self,dataframe,keywords_dataframe):
        self.df = dataframe.drop_duplicates(subset='name')
        self.k = keywords_dataframe
    
    def user_input(self):
        self.user_senti = float(input("Please give me your minimum acceptable score for user satisfaction. (between 0 to 10, one decimal)\n"))
        self.user_value = float(input("Please give me a minimum score for value for money. (between 0 to 10, one decimal)\n"))
        self.user_loc = input("Which one of these do you care about the most: Room, location, staff, bed, or breakfast?(lowercase)\n")
        result = self.df[(self.df['senti_normal'] >= self.user_senti) & (self.df['Value for money'] >= self.user_value)]
        result2 = result.loc[result['name'].isin (self.k[self.k['Keyword'] == self.user_loc]['name'])]
        top = result2.nlargest(3,'Score')       # 3 hotels for choice
        
        while len(top) == 0:
            print('')
            print ("I'm sorry, I cannot find any hotel that meets your requirements.\n")
            print ("Could you lower your expectation and try again?\n")
            self.user_senti = float(input("Please give me your minimum acceptable score for user satisfaction. (between 0 to 10, one decimal)\n"))
            self.user_value = float(input("Please give me a minimum score for value for money. (between 0 to 10, one decimal)\n"))
            self.user_loc = input("Which one of these do you care about the most: room, location, staff, bed, or breakfast?(lowercase)\n")
            result = self.df[(self.df['senti_normal'] >= self.user_senti) & (self.df['Value for money'] >= self.user_value)]
            result2 = result.loc[result['name'].isin (self.k[self.k['Keyword'] == self.user_loc]['name'])]
            top = result2.nlargest(3,'Score')
        self.top = top
        return top
        

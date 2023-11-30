import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

with open('./UserWiseReviews.txt') as reviewFile:
    reviews = reviewFile.readlines()
reviewFile.close()

reviewerIDlist = []
reviewLengthList = []
for line in reviews:
    reviewSplit = line.split("-")
    reviewerId = reviewSplit[0]
    reviewList = reviewSplit[1].strip().split("%#%")
    size = len(reviewList)
    reviewerIDlist.append(reviewerId)
    reviewLengthList.append(size-1)


print(len(reviewerIDlist))
print(len(reviewLengthList))
reviewPlot = pd.DataFrame({"ReviewerID":reviewerIDlist, "ReviewLength":reviewLengthList})

sns.lineplot(x="ReviewerID", y="ReviewLength", data=reviewPlot)
plt.xticks([])
plt.show()
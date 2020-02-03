install.packages("psych")

#Access the file

dataset = read.csv('C:\\Users\\hieut\\OneDrive - Oklahoma A and M System\\wdbc.csv', sep=",")

###Descriptive statistics
dim(dataset)
summary(dataset)
colnames(dataset)
str(dataset)
boxplot(dataset$perimeter_mean)

### Change diagnosis from categorical to numerical
dia1 = gsub("M",1,dataset$diagnosis)
dataset = data.frame(dataset, diag1)
dia2 = gsub("B",0,dataset$dia1)
dataset = data.frame(dataset, diag2)
### Remove old diagnosis rows and rename the column
dataset = subset(dataset, select=-c(diag1))]
names(dataset)[33] = "diag_status"
###M is now 1 and B is 0
##Plotting
plot(dataset$diag_status)
summary(dataset$diag_status)


###User mean, se, and worst variables as three separate regressions
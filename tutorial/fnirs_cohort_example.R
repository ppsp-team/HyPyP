library("ggplot2")

current_path <- rstudioapi::getActiveDocumentContext()$path 
base_path <- dirname(dirname(current_path))
csv_path <- paste0(base_path, "/data/results/fnirs_cohort_example.csv")

data <- read.csv(csv_path, header=TRUE, stringsAsFactors=FALSE)

data = within(data, {
  dyad <- as.factor(dyad)
  subject1 <- as.factor(subject1)
  subject2 <- as.factor(subject2)
  roi1 <- as.factor(roi1)
  roi2 <- as.factor(roi2)
  channel1 <- as.factor(channel1)
  channel2 <- as.factor(channel2)
  task <- as.factor(task)
  bin_time_range <- as.factor(bin_time_range)
  bin_period_range <- as.factor(bin_period_range)
  
  is_intra <- tolower(is_intra) == "true"
  is_shuffle <- tolower(is_shuffle) == "true"
})

data$roi_pair = paste(data$roi1, data$roi2, sep='-')

sel_not_intra = data$is_intra == FALSE
sel_not_shuffle = data$is_shuffle == FALSE
sel_not_na = sapply(data$coherence, is.na) == FALSE
data_inter <- subset(data, sel_not_intra & sel_not_shuffle & sel_not_na)

ggplot(data=data_inter, aes(x=channel1, y=channel2, fill=coherence)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Inter subject coherence')

ggplot(data=data_inter, aes(x=roi1, y=roi2, fill=coherence)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Inter subject coherence per region of interest')

ggplot(data=data_inter, aes(x=coherence, y=dyad, color=task)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Coherence')

ggplot(data=data_inter[data_inter$task!='baseline', ], aes(x=coherence, y=roi1)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  scale_y_discrete(labels = function(x) paste("Parent:", x)) +
  facet_wrap(.~ roi2, labeller = labeller(roi2 = function(x) paste("Child:", x))) +
  ggtitle('Zone pair coherence')

ggplot(data=data_inter, aes(x=coherence, y=bin_period_range, color=task)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Coherence per period range')














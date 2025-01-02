print("[+] Checking for installed packages")
local_installed_packages <- .packages(all.available = TRUE)

# Installing "arrow" from source might take a few minutes
required_packages <- c(
  "arrow",
  "ggplot2"
)

for (p in required_packages) {
  print(paste0("[*] Looking for package '", p, "': "))
  if (!(p %in% local_installed_packages)) {
    install.packages(p)
  }
}

library("arrow")
library("ggplot2")

current_path <- rstudioapi::getActiveDocumentContext()$path 
base_path <- dirname(dirname(current_path))
feather_path <- paste0(base_path, "/data/results/fnirs_cohort_example.feather")

data <- arrow::read_feather(feather_path)

data = within(data, {
  dyad <- as.factor(dyad)
  subject1 <- as.factor(subject1)
  subject2 <- as.factor(subject2)
  roi1 <- as.factor(roi1)
  roi2 <- as.factor(roi2)
  channel1 <- as.factor(channel1)
  channel2 <- as.factor(channel2)
  task <- as.factor(task)
})

data$roi_pair = paste(data$roi1, data$roi2, sep='-')

    
data_inter <- subset(data, data['is_intra'] == FALSE & data['is_shuffle'] == FALSE)
data_inter

ggplot(data=data_inter, aes(x=channel1, y=channel2, fill=coherence)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Inter subject coherence')

ggplot(data=data_inter, aes(x=dyad, y=coherence, color=task)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle('Coherence')

ggplot(data=data_inter, aes(x=roi1, y=coherence, color=task)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  scale_x_discrete(labels = function(x) paste("Parent:", x)) +
  facet_wrap(.~ roi2, labeller = labeller(roi2 = function(x) paste("Child:", x))) +
  ggtitle('Zone pair coherence')














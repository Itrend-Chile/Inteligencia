
library(ggplot2)
library(scatterpie)

topic_names_list <- c("Topic0", "Topic1", "Topic2", "Topic3", "Topic4", "Topic5", "Topic6", "Topic7")

# high resolution tiff image
tiff('scatterpie.png', units='in', width=12, height=9, res=500)
p <- ggplot() + 
  geom_scatterpie(aes(x= x_tsne, y=y_tsne, group=row_id, r=x_1_topic_probability), data=document_topic, cols=topic_names_list, color=NA, alpha=0.7) + 
  coord_equal() + 
  ggtitle("Scatterpie Chart") + 
  xlab("") + ylab("") + labs(subtitle="t-SNE Representation of Guided-LDA Topics Colored and Sized by Topic Probability") +
  scale_fill_manual(values=colors) + 
  theme_minimal() + 
  theme(text = element_text(color="white"),
        legend.position = "none",
        panel.background = element_rect(fill = "gray17", colour = "gray17"), 
        plot.background = element_rect(fill = "gray17"),
        panel.grid.major = element_line(colour = "gray25"),
        panel.grid.minor = element_line(colour = "gray25"),
        axis.text = element_text(color="white"))
# shut down graphics device
dev.off()

print(p)

# high resolution tiff image
tiff('scatterpie.png', units='in', width=12, height=9, res=500)
p <- ggplot() + 
  geom_scatterpie(aes(x= x_tsne, y=y_tsne, group=row_id, r=x_1_topic_probability), data=document_topic, cols=topic_names_list, color=NA, alpha=0.7) + 
  coord_equal() + 
  ggtitle("Scatterpie Chart") + 
  xlab("") + ylab("") + labs(subtitle="t-SNE Representation of LDA Topics") +
  scale_fill_manual(values=c("#CC0000", "#006600", "#669999", "#00CCCC", 
                             "#660099", "#CC0066", "#FF9999", "#FF9900"))+ 
  theme_minimal() + 
  theme(text = element_text(color="white"),
        legend.position = "none",
        panel.background = element_rect(fill = "gray17", colour = "gray17"), 
        plot.background = element_rect(fill = "gray17"),
        panel.grid.major = element_line(colour = "gray25"),
        panel.grid.minor = element_line(colour = "gray25"),
        axis.text = element_text(color="white"))
# shut down graphics device
dev.off()

print(p)
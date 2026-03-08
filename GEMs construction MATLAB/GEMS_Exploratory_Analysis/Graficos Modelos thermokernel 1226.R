# === Librerías ===
if(!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
library(ggplot2)

if(!require(scales)) install.packages("scales", dependencies=TRUE)
library(scales)

# === Leer CSV ===
df <- read.csv("/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Modelos cancer/GEMs cancer/Models/Resumen_Modelos_ThermoKernel.csv")

# Colores profesionales
color_reacciones <- "#1f77b4"  # azul
color_similitud <- "#2ca02c"   # verde

# -----------------------------
# Número de reacciones por modelo (ordenado de menor a mayor)
# -----------------------------
ggplot(df, aes(x = reorder(Modelo, NumReacciones), y = NumReacciones)) +
  geom_col(fill = color_reacciones, width=0.7) +
  labs(title="Número de reacciones por modelo (menor a mayor)", x='Modelos', y="Número de reacciones") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(size=13, face="bold", family="Times"),
        axis.title = element_text(size=11, family="Times"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        text = element_text(family="Times"))

# -----------------------------
# Similaridad promedio por modelo (ordenado de menor a mayor)
# -----------------------------
ggplot(df, aes(x = reorder(Modelo, SimilaridadPromedio), y = SimilaridadPromedio)) +
  geom_col(fill = color_similitud, width=0.7) +
  labs(title="Similaridad promedio por modelo (menor a mayor)", x='Modelos', y="Similaridad promedio") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(size=13, face="bold", family="Times"),
        axis.title = element_text(size=11, family="Times"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        text = element_text(family="Times"))

# === Histograma: Número de reacciones con línea de promedio ===
ggplot(df, aes(x = NumReacciones)) +
  geom_histogram(aes(y=..density..), bins=15, fill="lightgreen", color="black") +
  geom_density(color="darkgreen", size=1) +
  geom_vline(aes(xintercept=mean(NumReacciones)), color="red", linetype="dashed", size=1) +
  labs(title="Distribución del número de reacciones", x="Número de reacciones", y="Densidad") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(size=13, face="bold", family="Times"),
        axis.title = element_text(size=11, family="Times"),
        text = element_text(family="Times"))

# === Histograma: Similaridad promedio con línea de promedio ===
ggplot(df, aes(x = SimilaridadPromedio)) +
  geom_histogram(aes(y=..density..), bins=15, fill="lightcoral", color="black") +
  geom_density(color="darkred", size=1) +
  geom_vline(aes(xintercept=mean(SimilaridadPromedio)), color="blue", linetype="dashed", size=1) +
  labs(title="Distribución de similitudes promedio", x="Similitud promedio", y="Densidad") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(size=13, face="bold", family="Times"),
        axis.title = element_text(size=11, family="Times"),
        text = element_text(family="Times"))

# === Scatter plot: NumReacciones vs SimilaridadPromedio ===
ggplot(df, aes(x = NumReacciones, y = SimilaridadPromedio, label=Modelo)) +
  geom_point(aes(color=SimilaridadPromedio, size=NumReacciones)) +
  scale_color_gradient(low="lightblue", high="darkblue") +
  labs(title="Relación entre número de reacciones y similitud promedio", x="Número de reacciones", y="Similitud promedio") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(size=13, face="bold", family="Times"),
        axis.title = element_text(size=11, family="Times"),
        text = element_text(family="Times"))

from utils import print_header, print_step, print_footer
import pandas as pd
import mygene
import os

class XomicsFilesCreator:
    def __init__(self, input_path: str, output_path: str):
        self._validate_input_path(input_path)
        self._validate_output_path(output_path)
        self.input_path = input_path
        self.output_path = output_path

    def _validate_input_path(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The input path '{input_path}' does not exist.")

    def _validate_output_path(self, output_path: str):
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"The output path '{output_path}' does not exist.")

    def run_from_genomics(self):
        print_header()
        print_step("Loading data")

        # Cargar datos de expresión (índice: Ensembl con versión)
        data = pd.read_csv(self.input_path, delimiter="\t", index_col=0)
        genes_with_version = data.index.to_list()
        genes = [g.split('.')[0] for g in genes_with_version]

        print_step("Mapping Ensembl to Entrez gene IDs using MyGene.info")

        # Obtener IDs Entrez con mygene
        mg = mygene.MyGeneInfo()
        entries = mg.querymany(genes, scopes="ensembl.gene", fields="entrezgene", species="human", as_dataframe=True)

        # Quitar genes sin mapeo válido
        entries = entries[entries['notfound'].isna() | (entries['notfound'] == False)]
        entrez_map = entries['entrezgene'].to_dict()

        # Mapear Ensembl con versión → Entrez
        entrez_ids = [entrez_map.get(g.split('.')[0], None) for g in genes_with_version]

        # Filtrar genes sin entrez_id
        valid_idx = [i for i, eid in enumerate(entrez_ids) if eid is not None]
        genes_with_version = [genes_with_version[i] for i in valid_idx]
        entrez_ids = [entrez_ids[i] for i in valid_idx]
        data = data.iloc[valid_idx]

        # Validar
        assert len(entrez_ids) == data.shape[0], \
            f"Error: Length mismatch entrez_ids({len(entrez_ids)}), expression rows({data.shape[0]})"

        print_step("Generating Xomics files")

        for sample in data.columns:
            sample_expression = data[sample].values

            df_sample = pd.DataFrame({
                "genes": entrez_ids,
                "expVal": sample_expression
            })

            df_sample = df_sample.dropna(subset=["genes"])
            df_sample = df_sample[df_sample["expVal"] != 0]

            out_path = os.path.join(self.output_path, f"{sample}_Xomics.txt")
            df_sample.to_csv(out_path, index=False)

        print_footer()


# Ejecutar
input_path = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Modelos cancer/TCGA-BRCA_log_FPKM.tsv"
output_path = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Modelos cancer/Gems2"

generator = XomicsFilesCreator(input_path, output_path)
generator.run_from_genomics()

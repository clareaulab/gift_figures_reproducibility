library(biomaRt)

# use Ensembl GRCh38 (human)
mart <- useEnsembl(
  biomart = "ensembl",
  dataset = "hsapiens_gene_ensembl",
  version = 110  # pick a release known to use GRCh38
)

mane <- getBM(
  attributes = c(
    "ensembl_gene_id",
    "ensembl_transcript_id",
    "external_gene_name",
    "transcript_mane_select"
  ),
  mart = mart
)
mane <- subset(mane, transcript_mane_select != "")

coords <- getBM(
  attributes = c(
    "ensembl_transcript_id",
    "transcript_length",
    "cdna_coding_start",
    "cdna_coding_end"
  ),
  mart = mart
)

transcript_data <- merge(mane, coords, by = "ensembl_transcript_id")

write.table(
  transcript_data,
  file = "../data/MANE_Select_transcript_lengths.tsv",
  sep = "\t", row.names = FALSE, quote = FALSE
)

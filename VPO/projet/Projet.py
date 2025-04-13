import tkinter as tk
import tkinter.simpledialog as simpledialog

class FiltreDialog(simpledialog.Dialog):
    def __init__(self, parent, titre):
        self.titre = titre
        super().__init__(parent)

    def body(self, master):
        self.type_filtre = tk.StringVar()
        self.taille_filtre_rows = tk.StringVar()
        self.taille_filtre_cols = tk.StringVar()
        self.cutoff_frequency = tk.StringVar()

        tk.Label(master, text="Sélection du filtre passe-{}".format(self.titre)).pack()

        cadre_type_filtre = tk.Frame(master)
        cadre_type_filtre.pack(pady=10)
        tk.Label(cadre_type_filtre, text="Choisir le type de filtre :").pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Butterworth", variable=self.type_filtre, value="butterworth").pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Gaussien", variable=self.type_filtre, value="gaussien").pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Idéal", variable=self.type_filtre, value="ideal").pack(side="left", padx=5)

        cadre_taille_filtre = tk.Frame(master)
        cadre_taille_filtre.pack(pady=10)
        tk.Label(cadre_taille_filtre, text="Taille du filtre (rows x cols) :").pack(side="left", padx=5)
        self.entry_cols = tk.Entry(cadre_taille_filtre, width=20)
        self.entry_cols.pack(side="left", padx=5)
        tk.Label(cadre_taille_filtre, text="x").pack(side="left")
        self.entry_rows = tk.Entry(cadre_taille_filtre, width=20)
        self.entry_rows.pack(side="left", padx=5)

        cadre_cutoff_frequency = tk.Frame(master)
        cadre_cutoff_frequency.pack(pady=10)
        tk.Label(cadre_cutoff_frequency, text="Fréquence de coupure :").pack(side="left", padx=5)
        tk.Entry(cadre_cutoff_frequency, textvariable=self.cutoff_frequency, width=10).pack(side="left", padx=5)
        tk.Label(cadre_cutoff_frequency, text="Hz").pack(side="left")

    def apply(self):
        self.result = {
            "type_filtre": self.type_filtre.get(),
            "taille_filtre_rows": self.entry_rows.get(),
            "taille_filtre_cols": self.entry_cols.get(),
            "cutoff_frequency": self.cutoff_frequency.get()
        }

# Utilisation :
root = tk.Tk()
dialog = FiltreDialog(root, titre="haut")
print("Résultat:", dialog.result)
root.mainloop()

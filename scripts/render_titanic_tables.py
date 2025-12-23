from datafiller.datasets import load_titanic
from datafiller import MultivariateImputer, ExtremeLearningMachine


def main() -> None:
    df = load_titanic()
    imputer = MultivariateImputer(regressor=ExtremeLearningMachine())
    df_imputed = imputer(df)
    df.head(15).to_markdown("docs/_static/titanic_head.md", index=False)
    df_imputed.head(15).to_markdown("docs/_static/titanic_imputed_head.md", index=False)


if __name__ == "__main__":
    main()

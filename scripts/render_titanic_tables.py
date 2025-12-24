from datafiller import ExtremeLearningMachine, MultivariateImputer
from datafiller.datasets import load_titanic


def main() -> None:
    df = load_titanic()
    imputer = MultivariateImputer(regressor=ExtremeLearningMachine())
    df_imputed = imputer(df)
    df.to_csv("docs/_static/titanic.csv", index=False)
    df_imputed.to_csv("docs/_static/titanic_imputed.csv", index=False)


if __name__ == "__main__":
    main()

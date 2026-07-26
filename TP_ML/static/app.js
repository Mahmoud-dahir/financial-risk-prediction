document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("#credit-form");
    const submitButton = document.querySelector("#submit-button");
    const exampleButton = document.querySelector("#example-button");
    const resetButton = document.querySelector("#reset-button");
    const apiStatus = document.querySelector("#api-status");

    const resultEmpty = document.querySelector("#result-empty");
    const resultContent = document.querySelector("#result-content");
    const errorBox = document.querySelector("#error-box");
    const errorMessage = document.querySelector("#error-message");

    const decisionCard = document.querySelector("#decision-card");
    const decisionLabel = document.querySelector("#decision-label");
    const decisionDescription = document.querySelector("#decision-description");
    const predictionValue = document.querySelector("#prediction-value");
    const probabilityZero = document.querySelector("#probability-zero");
    const probabilityOne = document.querySelector("#probability-one");
    const scoreValue = document.querySelector("#score-value");
    const scoreFill = document.querySelector("#score-fill");

    const fields = [
        "Seniority",
        "Home",
        "Time",
        "Age",
        "Marital",
        "Records",
        "Job",
        "Expenses",
        "Income",
        "Assets",
        "Debt",
        "Amount",
        "Price"
    ];

    const exampleValues = {
        Seniority: 9,
        Home: 1,
        Time: 60,
        Age: 30,
        Marital: 0,
        Records: 1,
        Job: 1,
        Expenses: 73,
        Income: 129,
        Assets: 0,
        Debt: 0,
        Amount: 800,
        Price: 846
    };

    function setApiStatus(state, text) {
        apiStatus.classList.remove("online", "offline");

        if (state) {
            apiStatus.classList.add(state);
        }

        apiStatus.lastChild.textContent = ` ${text}`;
    }

    async function checkApiHealth() {
        try {
            const response = await fetch("/health", {
                headers: {
                    Accept: "application/json"
                }
            });

            if (!response.ok) {
                throw new Error(`Réponse HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.model_loaded === true) {
                setApiStatus("online", "API disponible");
            } else {
                setApiStatus("offline", "Modèle non chargé");
            }
        } catch (error) {
            setApiStatus("offline", "API indisponible");
        }
    }

    function formatPercent(value) {
        if (
            value === null ||
            value === undefined ||
            Number.isNaN(Number(value))
        ) {
            return "Non disponible";
        }

        return `${(Number(value) * 100).toLocaleString("fr-FR", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })} %`;
    }

    function readPayload() {
        return Object.fromEntries(
            fields.map((fieldName) => {
                const input = document.querySelector(`#${fieldName}`);

                return [
                    fieldName,
                    Number(input.value)
                ];
            })
        );
    }

    function setLoading(isLoading) {
        submitButton.disabled = isLoading;
        submitButton.classList.toggle("loading", isLoading);
    }

    function clearMessages() {
        errorBox.hidden = true;
        errorMessage.textContent = "";
    }

    function showError(message) {
        resultEmpty.hidden = true;
        resultContent.hidden = true;
        errorMessage.textContent = message;
        errorBox.hidden = false;
    }

    function showResult(data) {
        const prediction = Number(data.prediction);
        const probaOne = data.proba_classe_1;

        const scorePercent =
            probaOne === null || probaOne === undefined
                ? prediction * 100
                : Math.max(
                    0,
                    Math.min(100, Number(probaOne) * 100)
                );

        decisionCard.classList.toggle(
            "class-zero",
            prediction === 0
        );

        decisionLabel.textContent = `Classe ${prediction} prédite`;

        decisionDescription.textContent =
            `Le modèle associe ce dossier à la catégorie ${prediction} définie dans le jeu de données.`;

        predictionValue.textContent = String(prediction);
        probabilityZero.textContent = formatPercent(
            data.proba_classe_0
        );
        probabilityOne.textContent = formatPercent(
            data.proba_classe_1
        );

        scoreValue.textContent =
            `${scorePercent.toLocaleString("fr-FR", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })} %`;

        scoreFill.style.width = "0%";

        window.requestAnimationFrame(() => {
            scoreFill.style.width = `${scorePercent}%`;
        });

        errorBox.hidden = true;
        resultEmpty.hidden = true;
        resultContent.hidden = false;
    }

    async function submitCreditRequest(event) {
        event.preventDefault();
        clearMessages();

        if (!form.reportValidity()) {
            return;
        }

        const payload = readPayload();

        setLoading(true);

        try {
            const response = await fetch("/predict/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                let detail =
                    `Le serveur a répondu avec le code ${response.status}.`;

                try {
                    const errorData = await response.json();

                    if (errorData.detail) {
                        detail =
                            typeof errorData.detail === "string"
                                ? errorData.detail
                                : JSON.stringify(errorData.detail);
                    }
                } catch (parseError) {
                    // On conserve le message HTTP.
                }

                throw new Error(detail);
            }

            const data = await response.json();

            showResult(data);
        } catch (error) {
            showError(
                error instanceof Error
                    ? error.message
                    : "Une erreur inconnue est survenue."
            );
        } finally {
            setLoading(false);
        }
    }

    function loadExample() {
        Object.entries(exampleValues).forEach(
            ([fieldName, value]) => {
                const input =
                    document.querySelector(`#${fieldName}`);

                input.value = String(value);
            }
        );

        clearMessages();
        resultContent.hidden = true;
        resultEmpty.hidden = false;

        document.querySelector("#Seniority").focus();
    }

    function resetResult() {
        window.setTimeout(() => {
            clearMessages();
            resultContent.hidden = true;
            resultEmpty.hidden = false;
            scoreFill.style.width = "0%";
        }, 0);
    }

    form.addEventListener(
        "submit",
        submitCreditRequest
    );

    exampleButton.addEventListener(
        "click",
        loadExample
    );

    resetButton.addEventListener(
        "click",
        resetResult
    );

    checkApiHealth();
});

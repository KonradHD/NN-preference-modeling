from enum import Enum

class LossParameters(Enum):
    COP_WEIGHT = "cop_weight",
    FIT_WEIGHT = "fit_weight",
    STAB_WEIGHT =  "stab_weight",
    MARGIN = "margin", 
    NORMALIZE = "normalize", 
    P =   "p",
    COP_MODE = "cop_mode",
    STAB_MODE = "stab_mode"


LossParametersMapper: dict[LossParameters, str] = {
    LossParameters.COP_WEIGHT: "Waga członu COP",
    LossParameters.FIT_WEIGHT: "Waga członu FIT",
    LossParameters.STAB_WEIGHT: "Waga członu STAB",
    LossParameters.MARGIN: "Wartość marginesu błędu COP",
    LossParameters.NORMALIZE: "Czy członu funkcji straty są normalizowane",
    LossParameters.P: "Wartość współczynnik p",
    LossParameters.COP_MODE: "Tryb COP",
    LossParameters.STAB_MODE: "Tryb STAB"
}



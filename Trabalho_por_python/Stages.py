from dataclasses import dataclass
from typing import List, Tuple, Optional, Tuple, Dict
import bisect, math




def mix_viscosity_cP(z_light: float,
                     mu_light_cP: float,
                     mu_heavy_cP: float,
                     rule: str = "arrhenius") -> float:
    """
    Mistura de viscosidade do líquido da alimentação (cP).
    z_light: fração molar do componente MAIS volátil na alimentação (sua feed.z)
    rule:
      - "arrhenius" -> ln(mu_mix) = z ln(mu_L) + (1-z) ln(mu_H)  [recomendado para mu]
      - "linear"    -> mu_mix = z*mu_L + (1-z)*mu_H
    """
    z = max(0.0, min(1.0, z_light))
    if rule == "linear":
        return z*mu_light_cP + (1.0 - z)*mu_heavy_cP
    # padrão: Arrhenius (log-mixing)
    import math
    return math.exp(z*math.log(mu_light_cP) + (1.0 - z)*math.log(mu_heavy_cP))



def oconnell_overall_efficiency(alpha: float, muF_cP: float,
                                clip: tuple[float, float] = (0.05, 1.0)) -> float:
    """
    Eficiência global de prato (fração 0–1) via O'Connell:
        η_G(%) = 49.2 * (α * μ_F)^(-0.245)
    μ_F em cP; α = volatilidade relativa dos componentes chave.
    'clip' limita a eficiência a um intervalo físico (padrão [5%, 100%]).
    """
    import math
    eta_percent = 49.2 * (alpha * muF_cP) ** (-0.245)
    eta = eta_percent / 100.0
    return max(clip[0], min(eta, clip[1]))


def real_stages_from_efficiency(N_theoretical_total: int,
                                E_overall: float,
                                count_total_condenser: bool = True,
                                count_reboiler: bool = False) -> dict:
    """
    Converte estágios teóricos -> reais com eficiência global E_overall.
    Por padrão, conta condensador total como estágio e NÃO conta reboiler.
    """
    import math
    aux = (1 if count_total_condenser else 0) + (1 if count_reboiler else 0)
    N_teor_sem_aux = max(0, N_theoretical_total - aux)
    N_real_sem_aux = math.ceil(N_teor_sem_aux / E_overall)
    N_real_total   = N_real_sem_aux + aux
    return {
        "E_overall": E_overall,
        "N_teor_total": N_theoretical_total,
        "N_real_sem_aux": N_real_sem_aux,
        "N_real_total": N_real_total
    }


# =========================================
# ANTOINE + Mistura Ideal (Raoult)
# =========================================


MMHG_TO_BAR = 1.01325 / 760.0 



@dataclass
class AntoineComp:
    name: str
    A: float
    B: float
    C: float
    Tmin_C: Optional[float] = None
    Tmax_C: Optional[float] = None

    def Psat_bar(self, T_C: float) -> float:
        """P^sat em bar usando Antoine (P em mmHg -> bar)."""
        P_mmHg = 10.0**(self.A - self.B/(T_C + self.C))
        return P_mmHg * MMHG_TO_BAR

    def Tb_at_P(self, P_bar: float) -> float:
        """T (°C) na qual P^sat = P_bar (inversão direta de Antoine)."""
        P_mmHg = P_bar / MMHG_TO_BAR
        return self.B / (self.A - math.log10(P_mmHg)) - self.C


@dataclass
class IdealAntoineMixture:
    comps: List[AntoineComp]
    P_bar: float = 1.01325  # padrão: ~1 atm

    # ---------- Auxiliar: K-values (ideal Raoult) ----------
    def K_values(self, T_C: float) -> List[float]:
        return [ci.Psat_bar(T_C) / self.P_bar for ci in self.comps]

    # ---------- Temperatura de bolha: Σ x_i P^sat_i(T) = P ----------
    def bubble_T(self, x: List[float], T_lo_C: Optional[float] = None, T_hi_C: Optional[float] = None,
                 iters: int = 80) -> float:
        if len(x) != len(self.comps):
            raise ValueError("x e comps com tamanhos incompatíveis.")
        if not math.isclose(sum(x), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            # normaliza por segurança
            s = sum(x); x = [xi/s for xi in x]

        # janela de busca: use Tbs dos puros como bracket
        tbs = [c.Tb_at_P(self.P_bar) for c in self.comps]
        lo = min(tbs) - 30.0 if T_lo_C is None else T_lo_C
        hi = max(tbs) + 30.0 if T_hi_C is None else T_hi_C

        def f(T):
            return sum(xi * ci.Psat_bar(T) for xi, ci in zip(x, self.comps)) - self.P_bar

        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            # tenta expandir janela
            lo2, hi2 = lo - 100.0, hi + 100.0
            f_lo2, f_hi2 = f(lo2), f(hi2)
            if f_lo2 * f_hi2 > 0:
                raise ValueError("bubble_T: não houve mudança de sinal no bracket; ajuste T_lo/T_hi.")
            lo, hi = lo2, hi2

        for _ in range(iters):
            mid = 0.5*(lo + hi)
            fm = f(mid)
            if f_lo * fm <= 0:
                hi, f_hi = mid, fm
            else:
                lo, f_lo = mid, fm
        return 0.5*(lo + hi)
    # Σ y_i * P / P^sat_i(T) = 1   (orvalho)
    def dew_T(self, y: List[float], T_lo_C: Optional[float] = None,
              T_hi_C: Optional[float] = None, iters: int = 80) -> float:
        if not math.isclose(sum(y), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            s = sum(y); y = [yi/s for yi in y]
        tbs = [c.Tb_at_P(self.P_bar) for c in self.comps]
        lo = min(tbs) - 30.0 if T_lo_C is None else T_lo_C
        hi = max(tbs) + 30.0 if T_hi_C is None else T_hi_C

        def g(T):
            return sum(yi * self.P_bar / ci.Psat_bar(T) for yi, ci in zip(y, self.comps)) - 1.0

        g_lo, g_hi = g(lo), g(hi)
        if g_lo * g_hi > 0:
            lo2, hi2 = lo - 100.0, hi + 100.0
            g_lo2, g_hi2 = g(lo2), g(hi2)
            if g_lo2 * g_hi2 > 0:
                raise ValueError("dew_T: ajuste T_lo/T_hi (sem mudança de sinal).")
            lo, hi = lo2, hi2

        for _ in range(iters):
            mid = 0.5*(lo + hi)
            gm = g(mid)
            if g_lo * gm <= 0:
                hi, g_hi = mid, gm
            else:
                lo, g_lo = mid, gm
        return 0.5*(lo + hi)

    def y_from_x_T(self, x: List[float], T_C: float) -> List[float]:
        K = self.K_values(T_C)
        num = [Ki*xi for Ki, xi in zip(K, x)]
        s = sum(num)
        return [ni/s for ni in num]

    def x_from_y_T(self, y: List[float], T_C: float) -> List[float]:
        K = self.K_values(T_C)
        num = [yi / Ki for yi, Ki in zip(y, K)]
        s = sum(num)
        return [ni/s for ni in num]

@dataclass
class PhaseEnthalpyTable1D:
    xs: List[float]            # nós (0..1) – x para líquido, y para vapor
    hs_kJ_per_kmol: List[float]# valores (kJ/kmol)

    def __post_init__(self):
        if len(self.xs) != len(self.hs_kJ_per_kmol):
            raise ValueError("xs e hs com tamanhos diferentes.")
        # ordena + remove duplicatas
        pts = sorted(zip(self.xs, self.hs_kJ_per_kmol), key=lambda p: p[0])
        xs, hs = [pts[0][0]], [pts[0][1]]
        for xi, hi in pts[1:]:
            if xi != xs[-1]:
                xs.append(xi); hs.append(hi)
        self.xs, self.hs_kJ_per_kmol = xs, hs
        self.xmin, self.xmax = xs[0], xs[-1]

    def __call__(self, x: float) -> float:
        # clamp
        x = min(max(x, self.xmin), self.xmax)
        k = bisect.bisect_right(self.xs, x) - 1
        k = max(0, min(k, len(self.xs)-2))
        x1, x2 = self.xs[k], self.xs[k+1]
        h1, h2 = self.hs_kJ_per_kmol[k], self.hs_kJ_per_kmol[k+1]
        t = 0.0 if x2==x1 else (x - x1)/(x2 - x1)
        return h1 + t*(h2 - h1)

@dataclass
class BinaryVLE:
    xy_points: Optional[List[Tuple[float, float]]] = None
    alpha: Optional[float] = None



    def __post_init__(self):
        if self.xy_points:
            pts = sorted(self.xy_points, key=lambda p: p[0])
            # remove duplicatas em x mantendo a 1ª
            x_tab, y_tab = [pts[0][0]], [pts[0][1]]
            for xi, yi in pts[1:]:
                if xi != x_tab[-1]:
                    x_tab.append(xi); y_tab.append(yi)
            self.x_tab, self.y_tab = x_tab, y_tab
            self.x_min, self.x_max = self.x_tab[0], self.x_tab[-1]
        elif self.alpha is None:
            raise ValueError("Provide xy_points or alpha.")

    # ---------- ALPHA ----------


    @staticmethod
    def alpha_from_pair(x: float, y: float) -> float:
        eps = 1e-12
        if x <= eps or x >= 1-eps or y <= eps or y >= 1-eps:
            return math.nan
        return (y*(1.0 - x)) / (x*(1.0 - y))

    def alpha_pointwise(self) -> List[float]:
        if not self.xy_points:
            return [self.alpha] if self.alpha is not None else []
        return [self.alpha_from_pair(xi, yi) for xi, yi in zip(self.x_tab, self.y_tab)]

    def alpha_constant(self, method: str = "geom") -> float:
        vals = [a for a in self.alpha_pointwise() if a == a and a > 0]  # filtra NaN/neg
        if not vals:
            return math.nan
        if method == "median":
            s = sorted(vals); m = len(s)//2
            return s[m] if len(s)%2 else 0.5*(s[m-1]+s[m])
        # média geométrica
        return math.exp(sum(math.log(a) for a in vals)/len(vals))
    



    # ---------- VLE ----------
    def y_of_x(self, x: float) -> float:
        if self.xy_points:
            x = min(max(x, self.x_min), self.x_max)
            k = bisect.bisect_right(self.x_tab, x) - 1
            k = max(0, min(k, len(self.x_tab) - 2))
            x1, y1 = self.x_tab[k], self.y_tab[k]
            x2, y2 = self.x_tab[k+1], self.y_tab[k+1]
            dx = x2 - x1 if x2 > x1 else 1e-12
            t = (x - x1) / dx
            return y1 + t*(y2 - y1)
        else:
            a = self.alpha
            return (a*x) / (1.0 + (a - 1.0)*x)

    def x_of_y(self, y: float) -> float:
        y = min(max(y, 0.0), 1.0)
        if self.xy_points:
            lo, hi = self.x_min, self.x_max
            for _ in range(60):
                mid = 0.5*(lo+hi)
                ym = self.y_of_x(mid)
                if ym > y: hi = mid
                else:      lo = mid
            return 0.5*(lo+hi)
        else:
            a = self.alpha
            return y / (a - (a - 1.0)*y)
        
@dataclass
class Alimentacao:
    F:float # kmol/h
    z:float # fração moldar do mais volatil
    q:float # fração líquida alimentada (q=1: líquido saturado)

@dataclass
class ColunaMT:
    vle: BinaryVLE
    feed: Alimentacao
    mixture: Optional["IdealAntoineMixture"] = None  # <- adicionado
    thermo: Optional["IdealThermo"] = None  
    hL_of_x: Optional[PhaseEnthalpyTable1D] = None
    hV_of_y: Optional[PhaseEnthalpyTable1D] = None
    _last_Rmin: Optional[float] = None               # cache opcional

    def set_enthalpy_tables(self,
                            hL_table: PhaseEnthalpyTable1D,
                            hV_table: PhaseEnthalpyTable1D) -> None:
        """Define h^L(x) e h^V(y) em kJ/kmol para uso prioritário nos deveres térmicos."""
        self.hL_of_x = hL_table
        self.hV_of_y = hV_table

    # q-line: y = (q/(q-1)) x - zF/(q-1). Para q=1, é vertical em x=zF.
    def _y_qline(self, x: float) -> float:
        q, zF = self.feed.q, self.feed.z
        return (q/(q-1.0))*x - zF/(q-1.0)

    def calc_Rmin(self, xD: float) -> float:
        """
        Calcula R_min pelo pinch na interseção Equilíbrio = q-line.
        Para q=1: x* = zF (q-line vertical). Para q!=1: resolve y_eq(x*) = y_qline(x*).
        Retorna R_min = m/(1-m), m = (y* - xD)/(x* - xD).
        """
        q, zF = self.feed.q, self.feed.z

        # 1) achar x* do pinch
        if abs(q - 1.0) < 1e-10:
            x_star = zF
        else:
            # bisseção em janela segura
            x_lo = getattr(self.vle, "x_min", 0.0)
            x_hi = getattr(self.vle, "x_max", 1.0)

            def f(x): return self.vle.y_of_x(x) - self._y_qline(x)

            f_lo, f_hi = f(x_lo), f(x_hi)
            if f_lo * f_hi > 0:
                x_lo = max(0.0, zF - 0.49)
                x_hi = min(1.0, zF + 0.49)
                f_lo, f_hi = f(x_lo), f(x_hi)
                if f_lo * f_hi > 0:
                    raise ValueError("Não encontrei interseção equilíbrio x q-line; verifique VLE/q.")

            for _ in range(80):
                x_mid = 0.5*(x_lo + x_hi)
                f_mid = f(x_mid)
                if f_lo * f_mid <= 0:
                    x_hi, f_hi = x_mid, f_mid
                else:
                    x_lo, f_lo = x_mid, f_mid
            x_star = 0.5*(x_lo + x_hi)

        y_star = self.vle.y_of_x(x_star)

        # 2) inclinação mínima
        denom = (x_star - xD)
        if abs(denom) < 1e-14:
            raise ValueError("Pinch em x*=xD -> R_min infinito (especificação muito severa).")
        m = (y_star - xD) / denom
        if not (0.0 < m < 1.0):
            raise ValueError("R_min inviável (m fora de (0,1)). Verifique xD, zF, q e VLE.")

        Rmin = m / (1.0 - m)
        self._last_Rmin = Rmin
        return Rmin

    # --- Antoine/Mistura ---

    def set_mixture(self, mixture: "IdealAntoineMixture") -> None:
        """Atribui a mistura (Antoine) à coluna."""
        self.mixture = mixture

    def temperaturas_topo(self, prod: "Produto") -> Dict[str, float]:
        """
        Retorna T_bolha_topo e T_orvalho_topo (°C) na pressão da mistura.
        Usa x_top = [xD, 1-xD] e y_top em equilíbrio com T_bolha_topo.
        """
        if self.mixture is None:
            raise ValueError("Defina a mistura Antoine com set_mixture(...) antes.")
        x = [prod.xD, 1.0 - prod.xD]
        T_bub = self.mixture.bubble_T(x)
        y_eq = self.mixture.y_from_x_T(x, T_bub)
        T_dew = self.mixture.dew_T(y_eq)  # deve coincidir numericamente com T_bub
        return {
            "T_bolha_topo_C": T_bub,
            "T_orvalho_topo_C": T_dew,
            "y_topo_mais_volatil": y_eq[0]
        }

    def temperaturas_fundo(self, prod: "Produto") -> Dict[str, float]:
        """
        Retorna T_bolha_fundo e T_orvalho_fundo (°C) na pressão da mistura.
        Usa x_fundo = [xB, 1-xB] e y_fundo em equilíbrio com T_bolha_fundo.
        """
        if self.mixture is None:
            raise ValueError("Defina a mistura Antoine com set_mixture(...) antes.")
        xB = [prod.xB, 1.0 - prod.xB]
        T_bub = self.mixture.bubble_T(xB)
        y_eq = self.mixture.y_from_x_T(xB, T_bub)
        T_dew = self.mixture.dew_T(y_eq)
        return {
            "T_bolha_fundo_C": T_bub,
            "T_orvalho_fundo_C": T_dew,
            "y_fundo_mais_volatil": y_eq[0]
        }
    def set_thermo(self, thermo: "IdealThermo") -> None:
        """Atribui o modelo de entalpia ideal (Cp + Watson) à coluna."""
        # checagem simples: mesmo nº de componentes da mistura Antoine
        if self.mixture is not None and len(thermo.props) != len(self.mixture.comps):
            raise ValueError("thermo.props e mixture.comps com tamanhos diferentes.")
        self.thermo = thermo


    def condenser_summary(self, prod: "Produto",sign: str = "process", T_liq_out_C: Optional[float] = None) -> Dict[str, float]:
        """
        Entrada: vapor saturado no topo (T_top, y_top^eq)
        Saída:   líquido (xD) saturado (este método usa h^L(xD) tabulado; ignora subresfriamento)
        **Usa somente** hV_of_y e hL_of_x; lança erro se não houver tabela.
        """
        if self.mixture is None:
            raise ValueError("Defina mixture (Antoine) antes.")
        if self.hL_of_x is None or self.hV_of_y is None:
            raise ValueError("Defina tabelas de entalpia com set_enthalpy_tables(hL,hV).")

        # T e y do topo via Antoine
        top = self.temperaturas_topo(prod)
        T_in_vap = top["T_bolha_topo_C"]
        xD_vec = [prod.xD, 1.0 - prod.xD]
        y_top  = self.mixture.y_from_x_T(xD_vec, T_in_vap)

        # entalpias somente por TABELA (kJ/kmol)
        hV_in  = self.hV_of_y(y_top[0])  # y do mais volátil
        hL_out = self.hL_of_x(prod.xD)   # xD do mais volátil
        dh_process     = hV_in - hL_out
        if sign == "paper":
            dh = hL_out - hV_in
        else:
            dh= dh_process

        # vazão de vapor ao condensador (kmol/h)
        prod_res = prod
        if prod_res.R is None and prod_res.fator_R is not None:
            prod_res.resolve_R(self)
        flows = self.flows(prod_res)

        Q_kJh = flows.V_rect * dh
        P_kW  = Q_kJh / 3600.0
        return {
            "T_entrada_vapor_C": T_in_vap,
            "T_saida_liquido_C": T_in_vap,  # saturado (sem subresfriar)
            "hV_entrada_kJ_per_kmol": hV_in,
            "hL_saida_kJ_per_kmol":  hL_out,
            "delta_h_kJ_per_kmol":   dh,
            "delta_h_kJ_per_mol":    dh/1000.0,
            "Q_total_kJh": Q_kJh,
            "Potencia_kW": P_kW
        }

    def reboiler_summary(self, prod: "Produto") -> Dict[str, float]:
        """
        Entrada: líquido saturado no fundo (xB, T_bot)
        Saída:   vapor saturado (y_bot^eq, T_bot)
        **Usa somente** hL_of_x e hV_of_y; lança erro se não houver tabela.
        """
        if self.mixture is None:
            raise ValueError("Defina mixture (Antoine) antes.")
        if self.hL_of_x is None or self.hV_of_y is None:
            raise ValueError("Defina tabelas de entalpia com set_enthalpy_tables(hL,hV).")

        bot   = self.temperaturas_fundo(prod)
        T_bub = bot["T_bolha_fundo_C"]
        xBvec = [prod.xB, 1.0 - prod.xB]
        y_bot = self.mixture.y_from_x_T(xBvec, T_bub)

        hL_in  = self.hL_of_x(prod.xB)    # líquido de fundo
        hV_out = self.hV_of_y(y_bot[0])   # vapor gerado
        dh     = hV_out - hL_in

        prod_res = prod
        if prod_res.R is None and prod_res.fator_R is not None:
            prod_res.resolve_R(self)
        flows = self.flows(prod_res)

        Q_kJh = flows.V_strip * dh
        P_kW  = Q_kJh / 3600.0
        return {
            "T_entrada_liquido_C": T_bub,
            "T_saida_vapor_C":     T_bub,
            "hL_entrada_kJ_per_kmol": hL_in,
            "hV_saida_kJ_per_kmol":   hV_out,
            "delta_h_kJ_per_kmol":    dh,
            "delta_h_kJ_per_mol":     dh/1000.0,
            "Q_total_kJh": Q_kJh,
            "Potencia_kW": P_kW
        }
    
    def produto_por_fator(self, xD: float, xB: float, fator_R: float) -> "Produto":
        """R_op = fator_R * R_min; retorna Produto com R já resolvido."""
        Rmin = self.calc_Rmin(xD)
        return Produto(xD=xD, xB=xB, R=fator_R * Rmin)

    def print_Rmin(self, xD: float) -> float:
        Rmin = self.calc_Rmin(xD)
        print(f"R_min = {Rmin:.6f}")
        return Rmin

    def flows(self, prod: "Produto") -> "Fluxo":
        """
        Wrapper que garante R_op válido:
        - se prod.R for None e houver fator_R, resolve;
        - se ainda assim não houver R, erro.
        """
        if prod.R is None:
            if prod.fator_R is not None:
                prod.resolve_R(self)
            else:
                raise ValueError("Produto.R não resolvido. Informe R ou fator_R.")
        return compute_flows(self.feed, prod)
    # ---- Linhas de operação (usam SEMPRE R_op em prod.R) ----
    @staticmethod
    def _rectifying_line(x: float, prod: "Produto") -> float:
        return (prod.R/(prod.R+1.0))*x + prod.xD/(prod.R+1.0)

    @staticmethod
    def _stripping_line(x: float, flows: "Fluxo", prod: "Produto") -> float:
        return (flows.L_strip/flows.V_strip)*x - (flows.B/flows.V_strip)*prod.xB

    # ---- Contagem de estágios (analítico) ----
    def count_stages(self, prod: "Produto",
                     include_total_condenser_as_stage: bool = True,
                     x_tol: float = 1e-6, max_iter: int = 500) -> "MTResult":
        flows = self.flows(prod)  # garante R_op
        y = prod.xD
        stages: List[Tuple[float, float]] = []
        N_total = N_above = N_below = 0

        if include_total_condenser_as_stage:
            x = self.vle.x_of_y(y)
            stages.append((x, y))
            N_total += 1; N_above += 1
            y = self._rectifying_line(x, prod)

        for _ in range(max_iter):
            x = self.vle.x_of_y(y)
            stages.append((x, y))

            if x <= prod.xB + x_tol:
                break

            if self.feed.q == 1.0 and (flows.x_switch is not None) and (x > flows.x_switch - x_tol):
                y = self._rectifying_line(x, prod); N_above += 1
            else:
                y = self._stripping_line(x, flows, prod); N_below += 1

            N_total += 1

        return MTResult(N_total=N_total, N_above=N_above, N_below=N_below, stages=stages)

    def count_stages_from_factor(self, xD: float, xB: float, fator_R: float,
                                 include_total_condenser_as_stage: bool = True,
                                 x_tol: float = 1e-6, max_iter: int = 500) -> "MTResult":
        prod = self.produto_por_fator(xD=xD, xB=xB, fator_R=fator_R)  # já vem com R_op
        return self.count_stages(prod,
                                 include_total_condenser_as_stage=include_total_condenser_as_stage,
                                 x_tol=x_tol, max_iter=max_iter)
    # --------- α médio para O'Connell (pelos K(T)) ----------
    def alpha_at_avg_T(self, prod: "Produto") -> float:
        """
        Estima a volatilidade relativa α entre os dois componentes-chave
        na temperatura média (T_top + T_bottom)/2.
        Se houver 'mixture' (Antoine), usa K_i(T)=Psat_i(T)/P e α=K_light/K_heavy.
        Caso contrário, cai no α constante geométrico a partir do VLE (aprox.).
        """
        if self.mixture is None:
            # fallback (geom. mean ao longo da curva xy)
            return self.vle.alpha_constant(method="geom")

        # Temperaturas de topo e de fundo
        top = self.temperaturas_topo(prod)["T_bolha_topo_C"]
        bot = self.temperaturas_fundo(prod)["T_bolha_fundo_C"]
        T_avg = 0.5*(top + bot)

        # K-values na média — índice 0 = mais volátil
        K = self.mixture.K_values(T_avg)
        if len(K) != 2:
            raise ValueError("alpha_at_avg_T implementado para binário.")
        alpha = K[0] / K[1]
        return alpha
    # --------- Eficiência via O'Connell (genérica) ----------
    def oconnell_efficiency(self, prod: "Produto",
                             mu_light_cP: float,
                             mu_heavy_cP: float,
                             mix_rule: str = "arrhenius",
                             clip: tuple[float, float] = (0.05, 1.0)) -> float:
        """
        Calcula η_overall (fração 0–1) pela correlação de O'Connell.
        - μ_light_cP, μ_heavy_cP: viscosidades (cP) dos puros na condição da alimentação
        - mix_rule: 'arrhenius' (padrão) ou 'linear' para mistura de μ_F
        """
        # μ_F da alimentação (líquido): usa z = fração do MAIS volátil na feed
        muF = mix_viscosity_cP(self.feed.z, mu_light_cP, mu_heavy_cP, rule=mix_rule)

        # α médio na coluna
        alpha = self.alpha_at_avg_T(prod)

        # η overall (0–1)
        E = oconnell_overall_efficiency(alpha, muF, clip=clip)
        return E
    
    # --------- Contagem de estágios reais (atalho) ----------
    def count_real_stages_from_factor(self, xD: float, xB: float, fator_R: float,
                                      mu_light_cP: float, mu_heavy_cP: float,
                                      mix_rule: str = "arrhenius",
                                      include_total_condenser_as_stage: bool = True,
                                      count_reboiler_as_stage: bool = False) -> dict:
        """
        1) Conta estágios teóricos (McCabe–Thiele) para o fator_R dado.
        2) Calcula η via O'Connell.
        3) Converte para estágios reais.
        Retorna dict com: resultado teórico (MTResult), E_overall e N_real_total.
        """
        mt = self.count_stages_from_factor(xD, xB, fator_R,
                                           include_total_condenser_as_stage=include_total_condenser_as_stage)
        prod = self.produto_por_fator(xD, xB, fator_R)
        E = self.oconnell_efficiency(prod, mu_light_cP, mu_heavy_cP, mix_rule=mix_rule)
        conv = real_stages_from_efficiency(mt.N_total, E,
                                           count_total_condenser=include_total_condenser_as_stage,
                                           count_reboiler=count_reboiler_as_stage)
        return {"mt_result": mt, "E_overall": E, **conv}
    

@dataclass
class Produto:
    xD: float
    xB: float
    R: Optional[float] = None       # R_op (se já souber)
    fator_R: Optional[float] = None # fator * R_min (alternativo)

    def resolve_R(self, coluna: ColunaMT):
        """Se vier fator_R, calcula R_op internamente."""
        if self.R is None:
            if self.fator_R is None:
                raise ValueError("Informe R (operação) ou fator_R.")
            Rmin = coluna.calc_Rmin(self.xD)
            self.R = self.fator_R * Rmin

@dataclass
class Fluxo:
    D: float
    B: float
    L_rect: float
    V_rect: float
    L_strip: float
    V_strip: float
    x_switch: Optional[float]   # abscissa do prato de alimentação (para q=1, ~ zF)


def compute_flows(feed: Alimentacao, prod: Produto) -> Fluxo:

    # D, B por balanços totais
    D = feed.F*(feed.z - prod.xB)/(prod.xD - prod.xB)
    B = feed.F - D
    # Acima da carga
    L_rect = prod.R*D
    V_rect = L_rect + D
    # Abaixo da carga (q=1 -> L_strip = L_rect + F; V_strip = V_rect)
    L_strip = L_rect + feed.F if abs(feed.q - 1.0) < 1e-8 else L_rect + feed.q*feed.F
    V_strip = V_rect if abs(feed.q - 1.0) < 1e-8 else V_rect - (1.0 - feed.q)*feed.F
    # Troca de seção: para q=1, vertical em x = zF
    x_switch = feed.z if abs(feed.q - 1.0) < 1e-8 else None
    return Fluxo(D, B, L_rect, V_rect, L_strip, V_strip, x_switch)







@dataclass
class MTResult:
    N_total: int
    N_above: int
    N_below: int
    stages: List[Tuple[float, float]]  # [(x_i, y_i)] para depuração





@dataclass
class ComponentProps:
    name: str
    Tc_K: float                         # Temperatura crítica (K)
    Tb_C: float                         # Ebulição normal (°C)
    lambda_Tb_kJ_per_kmol: float        # λ na ebulição normal (kJ/kmol)
    CpL_kJ_per_kmolK: float             # Cp líquido (kJ/kmol-K)
    CpV_kJ_per_kmolK: float             # Cp vapor (kJ/kmol-K)

    def lambda_at_T(self, T_C: float) -> float:
        """Watson: λ(T) = λ(Tb) * ((1 - Tr) / (1 - Trb))^0.38,  Tr = T/Tc."""
        T_K  = T_C + 273.15
        Tb_K = self.Tb_C + 273.15
        Tr   = T_K  / self.Tc_K
        Trb  = Tb_K / self.Tc_K
        expo = 0.38
        # evita valores próximos de Tc
        num = max(1e-12, (1.0 - Tr))
        den = max(1e-12, (1.0 - Trb))
        return self.lambda_Tb_kJ_per_kmol * (num/den)**expo


@dataclass
class IdealThermo:
    """Entalpias ideais por mistura: h_L(T,x) e h_V^sat(T,y)."""
    props: List[ComponentProps]
    Tref_C: float = 0.0  # referência arbitrária para entalpias (cancela em Δh)

    def hL_mix(self, T_C: float, x: List[float]) -> float:
        """h_L misto (kJ/kmol de mistura) ~ soma x_i * CpL_i * (T - Tref)."""
        if not (abs(sum(x) - 1.0) < 1e-8): 
            s = sum(x); x = [xi/s for xi in x]
        dT = (T_C - self.Tref_C)
        return sum(xi * ci.CpL_kJ_per_kmolK * dT for xi, ci in zip(x, self.props))

    def hV_sat_mix(self, T_C: float, y: List[float]) -> float:
        """
        h_V^sat(T,y) (kJ/kmol de mistura) ~ soma y_i * [ h_L_i(T) + λ_i(T) ],
        com h_L_i(T) = CpL_i*(T - Tref).
        """
        if not (abs(sum(y) - 1.0) < 1e-8): 
            s = sum(y); y = [yi/s for yi in y]
        dT = (T_C - self.Tref_C)
        total = 0.0
        for yi, ci in zip(y, self.props):
            hL_i = ci.CpL_kJ_per_kmolK * dT
            lam  = ci.lambda_at_T(T_C)
            total += yi * (hL_i + lam)
        return total




import matplotlib.pyplot as plt

def plot_mccabe_thiele(col, prod, save_path,
                       include_diagonal=True, include_qline=True,
                       title=None):
    # Curva de equilíbrio (usa pontos tabulados se houver)
    if hasattr(col.vle, "x_tab"):
        xs_eq, ys_eq = col.vle.x_tab, col.vle.y_tab
    else:
        xs_eq = [i/300 for i in range(301)]
        ys_eq = [col.vle.y_of_x(xx) for xx in xs_eq]

    flows = col.flows(prod)

    # Retificação / Esgotamento / Diagonal
    xs = [i/300 for i in range(301)]
    y_rect  = [col._rectifying_line(xx, prod) for xx in xs]
    y_strip = [col._stripping_line(xx, flows, prod) for xx in xs]

    plt.figure(figsize=(7,7))
    plt.plot(xs_eq, ys_eq, label="Equilíbrio (y-x)")
    if include_diagonal:
        plt.plot([0,1], [0,1], label="Diagonal (y=x)")
    plt.plot(xs, y_rect,  label="Reta de retificação")
    plt.plot(xs, y_strip, label="Reta de esgotamento")

    # q-line
    if include_qline:
        q, zF = col.feed.q, col.feed.z
        if abs(q-1.0) < 1e-10:
            plt.plot([zF, zF], [0, 1], label="q-line (q=1)")
        else:
            def yq(x): return (q/(q-1.0))*x - zF/(q-1.0)
            xs_q = [i/300 for i in range(301)]
            plt.plot(xs_q, [yq(xx) for xx in xs_q], label="q-line")

    # Escadas (mesma lógica do seu count_stages, começando em (xD,xD))
    tol = 1e-6
    x_cur, y_cur = prod.xD, prod.xD
    # 1º patamar (condensador total como estágio)
    x_next = col.vle.x_of_y(y_cur)
    plt.plot([x_cur, x_next], [y_cur, y_cur])           # horizontal p/ equilíbrio
    y_line = col._rectifying_line(x_next, prod)
    plt.plot([x_next, x_next], [y_cur, y_line])         # vertical p/ linha de operação
    x_cur, y_cur = x_next, y_line

    # Demais patamares
    for _ in range(1000):
        x_next = col.vle.x_of_y(y_cur)
        plt.plot([x_cur, x_next], [y_cur, y_cur])       # horizontal
        if x_next <= prod.xB + tol:
            break
        if (col.feed.q == 1.0) and (flows.x_switch is not None) and (x_next > flows.x_switch - tol):
            y_line = col._rectifying_line(x_next, prod)
        else:
            y_line = col._stripping_line(x_next, flows, prod)
        plt.plot([x_next, x_next], [y_cur, y_line])     # vertical
        x_cur, y_cur = x_next, y_line

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("x (líquido)")
    plt.ylabel("y (vapor)")
    if title: plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
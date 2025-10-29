#analise_metricas.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class QualityMetrics:
    """
    Classe para cálculo e validação de métricas de qualidade de ajustes espectrais.
    
    Atributos:
        tolerance_center (float): Tolerância de erro para centro do pico (padrão: 2%)
        tolerance_amplitude (float): Tolerância de erro para amplitude (padrão: 10%)
        tolerance_sigma (float): Tolerância de erro para sigma (padrão: 15%)
        residuals_1sigma_min (float): % mínimo de resíduos dentro de 1σ (padrão: 65%)
        bias_max (float): Valor máximo aceitável para viés (padrão: 0.5)
    """
    
    def __init__(
        self,
        tolerance_center: float = 0.02,
        tolerance_amplitude: float = 0.10,
        tolerance_sigma: float = 0.15,
        residuals_1sigma_min: float = 0.65,
        bias_max: float = 0.5
    ):
        """Inicializa o validador com os requisitos do projeto."""
        self.tolerance_center = tolerance_center
        self.tolerance_amplitude = tolerance_amplitude
        self.tolerance_sigma = tolerance_sigma
        self.residuals_1sigma_min = residuals_1sigma_min
        self.bias_max = bias_max
        
        # Armazena resultados da última validação
        self.last_metrics = {}
        self.last_validation = {}
    
    
    def calculate_mse_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula Mean Squared Error (MSE) e Root Mean Squared Error (RMSE).
        
        Args:
            y_true: Array com valores verdadeiros (espectro original)
            y_pred: Array com valores preditos (modelo ajustado)
            
        Returns:
            Dicionário com 'mse' e 'rmse'
            
        Exemplo:
            >>> metrics = validator.calculate_mse_rmse(spectrum, fitted_model)
            >>> print(f"RMSE: {metrics['rmse']:.2f}")
        """

        if len(y_true) != len(y_pred):
            raise ValueError("Arrays devem ter o mesmo tamanho")
        
        # Calcula resíduos
        residuals = y_true - y_pred
        
        # MSE: média dos quadrados dos resíduos
        mse = np.mean(residuals**2)
        
        # RMSE: raiz do MSE
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse)
        }
    
    
    def calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o coeficiente de determinação R² (opcional).
        
        R² mede quanto da variância dos dados é explicada pelo modelo.
        R² = 1 → ajuste perfeito
        R² = 0 → modelo não explica nada
        R² < 0 → modelo pior que a média
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            
        Returns:
            Valor de R² entre -∞ e 1
        """
    
        ss_res = np.sum((y_true - y_pred)**2)  # Soma dos quadrados dos resíduos
        ss_tot = np.sum((y_true - np.mean(y_true))**2)  # Variância total
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    
    def analyze_residuals(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Analisa estatisticamente os resíduos do ajuste.
        
        Verifica:
        1. Média (bias) - deve ser ~0
        2. Desvio padrão
        3. % de pontos dentro de 1σ - deve ser ~68% (usar 65% como mínimo)
        4. Teste de normalidade (opcional)
        
        Args:
            residuals: Array com resíduos (y_true - y_pred)
            return_detailed: Se True, inclui teste de normalidade
            
        Returns:
            Dicionário com estatísticas dos resíduos
        """

        if len(residuals) == 0:
            raise ValueError("Array de resíduos está vazio")
        
        # 1. Calcula média (bias) - deve ser próximo de 0
        bias = np.mean(residuals)
        
        # 2. Calcula desvio padrão
        std = np.std(residuals, ddof=1)  # ddof=1 para amostra
        
        # 3. Calcula % de pontos dentro de ±1σ
        # Para distribuição normal: ~68%
        within_1sigma = np.sum(np.abs(residuals - bias) <= std) / len(residuals)
        
        # 4. Calcula % de pontos dentro de ±2σ (deve ser ~95%)
        within_2sigma = np.sum(np.abs(residuals - bias) <= 2*std) / len(residuals)
        
        result = {
            'bias': float(bias),
            'std': float(std),
            'within_1sigma': float(within_1sigma),
            'within_2sigma': float(within_2sigma),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'n_points': len(residuals)
        }
        return result
    
    
    def validate_peak_parameters(self, true_params: np.ndarray, fitted_params: np.ndarray,
                                  param_names: List[str] = None) -> Dict[str, Dict]:
        """
        Valida se os parâmetros recuperados estão dentro das tolerâncias.
        
        Para cada pico gaussiano, compara:
        - Centro (μ): erro relativo < 2%
        - Amplitude (A): erro relativo < 10%
        - Sigma (σ): erro relativo < 15%
        
        Args:
            true_params: Array 2D (n_peaks, 3) com parâmetros verdadeiros
                        Colunas: [amplitude, centro, sigma]
            fitted_params: Array 2D (n_peaks, 3) com parâmetros ajustados
                          Mesma estrutura que true_params
            param_names: Lista com nomes dos parâmetros (opcional)
                        Padrão: ['amplitude', 'centro', 'sigma']
            
        Returns:
            Dicionário com resultados da validação para cada pico
            
        Exemplo:
            >>> true = np.array([[1000, 512, 5], [800, 768, 4]])
            >>> fitted = np.array([[1020, 513, 5.1], [790, 770, 4.2]])
            >>> validation = validator.validate_peak_parameters(true, fitted)
            >>> print(validation['peak_0']['centro']['passed'])
        """
        if param_names is None:
            param_names = ['amplitude', 'centro', 'sigma']
        
        # Garante que são arrays 2D
        true_params = np.atleast_2d(true_params)
        fitted_params = np.atleast_2d(fitted_params)
        
        if true_params.shape != fitted_params.shape:
            raise ValueError("Arrays de parâmetros devem ter a mesma forma")
        
        n_peaks = true_params.shape[0]
        n_params = true_params.shape[1]
        
        # Define tolerâncias para cada tipo de parâmetro
        tolerances = {
            'amplitude': self.tolerance_amplitude,
            'centro': self.tolerance_center,
            'sigma': self.tolerance_sigma
        }
        
        # Resultados
        validation_results = {}
        
        # Para cada pico
        for i in range(n_peaks):
            peak_key = f'peak_{i}'
            validation_results[peak_key] = {}
            
            # Para cada parâmetro (amplitude, centro, sigma)
            for j in range(n_params):
                param_name = param_names[j] if j < len(param_names) else f'param_{j}'
                
                true_val = true_params[i, j]
                fitted_val = fitted_params[i, j]
                
                # Calcula erro absoluto e relativo
                abs_error = abs(fitted_val - true_val)
                
                # Evita divisão por zero
                if abs(true_val) < 1e-10:
                    rel_error = abs_error
                else:
                    rel_error = abs_error / abs(true_val)
                
                # Verifica se passou na tolerância
                tolerance = tolerances.get(param_name, 0.20)  # 20% default
                passed = rel_error < tolerance
                
                validation_results[peak_key][param_name] = {
                    'true': float(true_val),
                    'fitted': float(fitted_val),
                    'abs_error': float(abs_error),
                    'rel_error': float(rel_error),
                    'rel_error_percent': float(rel_error * 100),
                    'tolerance': float(tolerance),
                    'tolerance_percent': float(tolerance * 100),
                    'passed': bool(passed)
                }
        
        return validation_results
    
    
    def check_requirements(self, y_true: np.ndarray, y_pred: np.ndarray, true_params: Optional[np.ndarray] = None,
                            fitted_params: Optional[np.ndarray] = None, noise_level: Optional[float] = None) -> Dict[str, any]:
        """
        Verifica se todos os requisitos da seção 9 foram atendidos.
        
        Requisitos verificados:
        1. Parâmetros recuperados dentro das tolerâncias
        2. RMSE comparável ao nível de ruído
        3. Resíduos com média ~0 (|bias| < 0.5)
        4. ~65% dos resíduos dentro de 1σ
        
        Args:
            y_true: Espectro verdadeiro
            y_pred: Modelo ajustado
            true_params: Parâmetros verdadeiros dos picos (opcional)
            fitted_params: Parâmetros ajustados (opcional)
            noise_level: Nível de ruído esperado (opcional)
            
        Returns:
            Dicionário completo com todas as métricas e validações
        """
        results = {
            'global_metrics': {},
            'residual_analysis': {},
            'parameter_validation': {},
            'requirements_check': {}
        }
        
        # 1. Métricas globais
        mse_rmse = self.calculate_mse_rmse(y_true, y_pred)
        r2 = self.calculate_r_squared(y_true, y_pred)
        
        results['global_metrics'] = {
            **mse_rmse,
            'r_squared': r2
        }
        
        # 2. Análise de resíduos
        residuals = y_true - y_pred
        residual_stats = self.analyze_residuals(residuals)
        results['residual_analysis'] = residual_stats
        
        # 3. Validação de parâmetros (se fornecidos)
        if true_params is not None and fitted_params is not None:
            param_validation = self.validate_peak_parameters(true_params, fitted_params)
            results['parameter_validation'] = param_validation
        
        # 4. Verificação de requisitos
        req = results['requirements_check']
        
        # Requisito 1: Parâmetros dentro das tolerâncias
        if results['parameter_validation']:
            all_passed = True
            for peak_data in results['parameter_validation'].values():
                for param_data in peak_data.values():
                    if not param_data['passed']:
                        all_passed = False
                        break
                if not all_passed:
                    break
            
            req['parameters_within_tolerance'] = all_passed
        else:
            req['parameters_within_tolerance'] = None
        
        # Requisito 2: RMSE comparável ao ruído
        if noise_level is not None:
            # RMSE deve estar entre 0.5x e 2x do nível de ruído
            rmse = mse_rmse['rmse']
            req['rmse_comparable_to_noise'] = bool((0.5 * noise_level <= rmse <= 2.0 * noise_level))
            req['noise_level'] = noise_level
            req['rmse_noise_ratio'] = rmse / noise_level
        else:
            req['rmse_comparable_to_noise'] = None
        
        # Requisito 3: Bias próximo de zero
        bias = residual_stats['bias']
        req['bias_near_zero'] = abs(bias) < self.bias_max
        
        # Requisito 4: ~65% dos resíduos dentro de 1σ
        within_1sigma = residual_stats['within_1sigma']
        req['residuals_normal_distribution'] = within_1sigma >= self.residuals_1sigma_min
        
        # Resumo geral
        checks = [v for v in req.values() if v is not None and isinstance(v, bool)]
        req['all_requirements_met'] = all(checks) if checks else False
        req['n_requirements_checked'] = len(checks)
        req['n_requirements_passed'] = sum(checks)
        
        # Armazena para uso posterior
        self.last_metrics = results

        return results
    
    
    def generate_report(self, results: Optional[Dict] = None, print_report: bool = True) -> str:
        """
        Gera relatório formatado com os resultados da validação.
        
        Args:
            results: Dicionário com resultados (usa last_metrics se None)
            print_report: Se True, imprime no terminal
            detailed: Se True, inclui detalhes completos
            
        Returns:
            String com o relatório formatado
        """
        if results is None:
            results = self.last_metrics
        
        if not results:
            return "Nenhum resultado disponível. Execute check_requirements() primeiro."
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("RELATÓRIO DE VALIDAÇÃO DE QUALIDADE")
        lines.append("="*80 + "\n")
        
        # 1. Métricas Globais
        if 'global_metrics' in results:
            lines.append("1. MÉTRICAS GLOBAIS")
            lines.append("-" * 80)
            gm = results['global_metrics']
            lines.append(f"  MSE (Mean Squared Error):  {gm['mse']:>12.4f}")
            lines.append(f"  RMSE (Root MSE):           {gm['rmse']:>12.4f}")
            lines.append(f"  R² (Coef. Determinação):   {gm['r_squared']:>12.4f}")
            lines.append("")
        
        # 2. Análise de Resíduos
        if 'residual_analysis' in results:
            lines.append("2. ANÁLISE DE RESÍDUOS")
            lines.append("-" * 80)
            ra = results['residual_analysis']
            lines.append(f"  Média (Bias):              {ra['bias']:>12.4f}")
            lines.append(f"  Desvio Padrão:             {ra['std']:>12.4f}")
            lines.append(f"  % dentro de 1σ:            {ra['within_1sigma']:>11.1%}")
            lines.append(f"  % dentro de 2σ:            {ra['within_2sigma']:>11.1%}")
            lines.append(f"  Mínimo:                    {ra['min']:>12.4f}")
            lines.append(f"  Máximo:                    {ra['max']:>12.4f}")
            lines.append("")
        
        # 3. Validação de Parâmetros
        if 'parameter_validation' in results and results['parameter_validation']:
            lines.append("3. VALIDAÇÃO DE PARÂMETROS DOS PICOS")
            lines.append("-" * 80)
            
            pv = results['parameter_validation']
            
            for peak_name, peak_data in pv.items():
                lines.append(f"\n  {peak_name.upper().replace('_', ' ')}:")
                lines.append("  " + "-" * 76)
                
                # Cabeçalho da tabela
                lines.append(f"  {'Parâmetro':<12} {'Verdadeiro':>12} {'Ajustado':>12} "
                           f"{'Erro Rel.':>12} {'Tolerância':>12} {'Status':>8}")
                lines.append("  " + "-" * 76)
                
                for param_name, param_data in peak_data.items():
                    status = "✓ PASS" if param_data['passed'] else "✗ FAIL"
                    lines.append(
                        f"  {param_name.capitalize():<12} "
                        f"{param_data['true']:>12.4f} "
                        f"{param_data['fitted']:>12.4f} "
                        f"{param_data['rel_error_percent']:>11.2f}% "
                        f"{param_data['tolerance_percent']:>11.1f}% "
                        f"{status:>8}"
                    )
        
        # 4. Verificação de Requisitos
        if 'requirements_check' in results:
            lines.append("\n\n4. VERIFICAÇÃO DE REQUISITOS")
            lines.append("-" * 80)
            
            req = results['requirements_check']
            
            def format_check(value):
                if value is None:
                    return "○ N/A"
                return "✓ SIM" if value else "✗ NÃO"
            
            if req.get('parameters_within_tolerance') is not None:
                lines.append(f"  Parâmetros dentro das tolerâncias:     "
                           f"{format_check(req['parameters_within_tolerance'])}")
            
            if req.get('rmse_comparable_to_noise') is not None:
                lines.append(f"  RMSE comparável ao ruído:              "
                           f"{format_check(req['rmse_comparable_to_noise'])}")
                lines.append(f"    - Nível de ruído: {req['noise_level']:.4f}")
                lines.append(f"    - Razão RMSE/ruído: {req['rmse_noise_ratio']:.4f}")
            
            if req.get('bias_near_zero') is not None:
                lines.append(f"  Viés próximo de zero (|bias| < 0.5):   "
                           f"{format_check(req['bias_near_zero'])}")
            
            if req.get('residuals_normal_distribution') is not None:
                lines.append(f"  Resíduos normalmente distribuídos:     "
                           f"{format_check(req['residuals_normal_distribution'])}")
            
            lines.append("")
            lines.append("  " + "="*76)
            
            if req['all_requirements_met']:
                lines.append(f"  RESULTADO FINAL: ✓ TODOS OS REQUISITOS ATENDIDOS "
                           f"({req['n_requirements_passed']}/{req['n_requirements_checked']})")
            else:
                lines.append(f"  RESULTADO FINAL: ✗ ALGUNS REQUISITOS NÃO ATENDIDOS "
                           f"({req['n_requirements_passed']}/{req['n_requirements_checked']})")
            
            lines.append("  " + "="*76)
        
        lines.append("\n" + "="*80 + "\n")
        
        report = "\n".join(lines)
        
        if print_report:
            print(report)
        
        return report


# ============================================================================
# FUNÇÕES AUXILIARES PARA USO SIMPLIFICADO
# ============================================================================

def quick_validation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_params: Optional[np.ndarray] = None,
    fitted_params: Optional[np.ndarray] = None,
    noise_level: Optional[float] = None,
    print_report: bool = True
) -> Dict:
    """
    Função de conveniência para validação rápida.
    
    Exemplo de uso:
        >>> results = quick_validation(
        ...     y_true=spectrum,
        ...     y_pred=fitted_model,
        ...     true_params=true_peak_params,
        ...     fitted_params=fitted_peak_params,
        ...     noise_level=10.0
        ... )
    """
    validator = QualityMetrics()
    results = validator.check_requirements(
        y_true, y_pred, true_params, fitted_params, noise_level
    )
    validator.generate_report(results, print_report=print_report)
    return results
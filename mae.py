from datetime import datetime, timedelta
import re
import requests


_URL = 'https://www.mae.com.ar/reportes/api/General'


class PreciosMAE:
    """Engine de precios de MAE\n
    Parámetros:\n
        tgt_date -- (str o datetime) Fecha de formato ISO (ej. 'YYYY-MM-DD') o datetime
    """


    def __init__(self, tgt_date):
        if isinstance(tgt_date, datetime):
            tgt_date = tgt_date.strftime('%Y-%m-%d')

        try:
            str_date = tgt_date.replace('-', '')

            req1 = requests.get(_URL + '/LeerTabla/operacionesextracto/' + str_date + '/' + str_date)
            reg = req1.json()['data'][0]['Data'][-1]

            if reg['Fecha'] == tgt_date:
                id = reg['Id']
            else:
                raise RuntimeError('No hay datos para la fecha seleccionada.')

            req2 = requests.get(_URL + '/LeerDatosAdjuntos/operacionesextracto/' + id)
            raw_data = req2.text

            pattern = r'\s[A-Z0-9\s]{5}\s.{7}.{12}\s.{15}\s.{16}.{14}.{14}.{12}\s.{6}\s.{12}\s.{12}'
            registros = re.findall(pattern, raw_data)

            self._precios = {f'{r[1:6].strip()}-{r[12]}-{r[13]}': float(r[15:26].replace(',', '.')) * 100.0
                             for r in registros if float(r[15:26].replace(',', '.')) != 0.0}

        except TypeError:
            raise RuntimeError('Error en la consulta al sitio de MAE.')

    def get_available_tickers(self):
        """Retorna lista de activos disponibles"""
        return sorted(self._precios.keys())

    """Cosas a tener en cuenta a la hora de buscar los precios:
    - Los activos se negocian varias veces en pesos y dolares, por lo cual podes bajar ambos y sacar el TC implicito de trasacción lo que sería el MEP de cada activo.
    - Siempre es mejor tomar el T+2 para las transacciones porque es como expresamos la mayoría de las tasas. 
    - El codigo tira un error de el uso de apend, nada eso se puede cambiar pero es fiaca y no afecta
    - OJO el la moneda del activo se pone al final.

    """
    def get_price(self, ticker, sett_days, currency):
        """Retorna precios de MAE.\n
        Parámetros:\n
            ticker -- código del activo\n
            sett_days -- 0, 1, 2\n
            currency -- '$' / 'D'
        """
        tk = ticker + '-' + str(sett_days) + '-' + currency
        return self._precios.get(tk)



if __name__ == '__main__':

    mae = PreciosMAE('2022-12-22')
    print(mae.get_price('GD30', 0, '$'))
    print(mae.get_available_tickers())
    







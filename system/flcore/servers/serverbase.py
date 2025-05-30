import torch # type: ignore
import os
import numpy as np
import h5py # type: ignore
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        """
            Inicializa e configura a lista de clientes para o cenário de treinamento federado.

            Para cada cliente (de 0 até `self.num_clients - 1`), esta função:
            - Lê os dados de treinamento e teste específicos do cliente a partir do dataset.
            - Cria uma instância do cliente utilizando a classe passada em `clientObj`.
            - Define atributos do cliente, incluindo a quantidade de amostras de treino e teste,
                além dos indicadores de lentidão de treinamento e envio.
            - Adiciona o cliente criado à lista `self.clients`.

            Args:
                clientObj (class): A classe do cliente que será instanciada para cada cliente.
                    Deve possuir um construtor com a seguinte assinatura (pelo menos):
                        __init__(args, id, train_samples, test_samples, train_slow, send_slow)
                    onde:
                        - args: argumentos de configuração geral,
                        - id: identificador do cliente (int),
                        - train_samples: número de amostras de treino do cliente (int),
                        - test_samples: número de amostras de teste do cliente (int),
                        - train_slow: booleano indicando se o cliente treina lentamente,
                        - send_slow: booleano indicando se o cliente é lento ao enviar dados.

            Efeitos colaterais:
                - Modifica o atributo `self.clients`, preenchendo com as instâncias de clientes criadas.
            
            Dependências:
                - `self.num_clients` deve estar definido com o número total de clientes.
                - `self.train_slow_clients` e `self.send_slow_clients` são listas booleanas que indicam
                se cada cliente é lento no treino e no envio, respectivamente.
                - `self.dataset` deve conter os dados de todos os clientes.
                - `self.few_shot` é um parâmetro utilizado na leitura dos dados para configurar o few-shot learning.
                - Função `read_client_data(dataset, client_id, is_train, few_shot)` que retorna os dados do cliente.

            Exemplo:
                >>> set_clients(ClientClass)
                # Cria e adiciona os clientes do tipo ClientClass na lista self.clients
        """
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        """
        Seleciona aleatoriamente uma fração dos clientes para serem marcados como "lentos".

        Esta função gera uma lista booleana indicando quais clientes são considerados lentos
        (por exemplo, lentos no processamento ou comunicação), baseada na taxa `slow_rate`.
        O número de clientes lentos é dado pelo produto da taxa pelo total de clientes.
        A seleção é aleatória, e os demais clientes são marcados como rápidos (False).

        Args:
            slow_rate (float): Proporção (entre 0 e 1) dos clientes que devem ser selecionados como lentos.

        Returns:
            List[bool]: Lista com comprimento igual a `self.num_clients` onde:
                - True indica que o cliente naquela posição foi selecionado como lento.
                - False indica cliente rápido.

        Exemplo:
            >>> slow_clients = select_slow_clients(0.2)
            >>> sum(slow_clients)
            20  # se self.num_clients for 100, por exemplo.

        """
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        """
        Define quais clientes serão considerados lentos no treinamento e no envio de dados.

        Essa função utiliza o método `select_slow_clients` para selecionar aleatoriamente
        clientes lentos em dois aspectos diferentes:
        - Lentidão no treinamento (`train_slow_clients`).
        - Lentidão no envio de dados (`send_slow_clients`).

        A seleção é feita com base nas taxas configuradas `self.train_slow_rate` e
        `self.send_slow_rate`, que indicam a proporção de clientes lentos em cada categoria.

        Após a execução, os atributos:
        - `self.train_slow_clients`: lista booleana indicando quais clientes treinam lentamente.
        - `self.send_slow_clients`: lista booleana indicando quais clientes são lentos ao enviar dados.
        ficam atualizados.

        Não possui parâmetros de entrada nem retorno.

        Exemplo:
            >>> set_slow_clients()
            # Atualiza as listas self.train_slow_clients e self.send_slow_clients

        """
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        """
        Seleciona um subconjunto de clientes para participarem da rodada atual de treinamento.

        O número de clientes selecionados depende da configuração `random_join_ratio`:
        - Se `random_join_ratio` for True, o número de clientes que se juntam (`current_num_join_clients`)
            é sorteado aleatoriamente entre `num_join_clients` e `num_clients` (inclusive).
        - Caso contrário, `current_num_join_clients` é fixado em `num_join_clients`.

        Em seguida, seleciona-se aleatoriamente esse número de clientes da lista `self.clients`
        sem repetição.

        Returns:
            List[clientObj]: Lista com as instâncias dos clientes selecionados para participar
                            da rodada de treinamento atual.

        Efeitos colaterais:
            - Atualiza o atributo `self.current_num_join_clients` com o número de clientes selecionados.

        Exemplo:
            >>> selected = select_clients()
            >>> len(selected)
            10  # Se num_join_clients for 10 e random_join_ratio=False

        """
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        """
        Envia os parâmetros do modelo global para todos os clientes registrados.

        Para cada cliente na lista `self.clients`, esta função:
        - Mede o tempo gasto para enviar os parâmetros do modelo global ao cliente.
        - Atualiza o modelo do cliente com os parâmetros atuais (`client.set_parameters`).
        - Acumula estatísticas de custo temporal relacionadas ao envio do modelo,
            incrementando o número de rodadas (`num_rounds`) e somando o custo total de tempo.
        - O custo total de tempo considera o dobro do tempo medido para representar
            envio e recepção, conforme o critério adotado.

        Levanta:
            AssertionError: Se não houver clientes registrados em `self.clients`.

        Requisitos:
        - Cada cliente deve possuir os métodos `set_parameters` e o atributo `send_time_cost`,
            onde `send_time_cost` é um dicionário contendo as chaves 'num_rounds' e 'total_cost'.

        Exemplo:
            >>> send_models()
            # Atualiza os parâmetros do modelo de cada cliente e registra o custo de envio.

        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """
        Recebe os modelos treinados dos clientes selecionados que não foram descartados.

        O processo inclui:
        - Verificar se há clientes selecionados para receber modelos (assert).
        - Selecionar aleatoriamente um subconjunto de clientes ativos, considerando a taxa de queda (`client_drop_rate`).
        - Para cada cliente ativo, calcula o custo médio total de tempo do cliente, somando
            o custo médio de treinamento e de envio, tratando possível divisão por zero.
        - Filtra clientes cujo custo de tempo seja menor ou igual ao limite definido (`time_threthold`).
        - Para os clientes que passam no filtro:
            - Acumula o total de amostras de treino (`tot_samples`).
            - Armazena o ID do cliente, o número de amostras e o modelo treinado nas listas correspondentes.
        - Normaliza os pesos dos clientes (`uploaded_weights`) dividindo pelo total de amostras para ponderação.

        Levanta:
            AssertionError: Se `self.selected_clients` estiver vazio.

        Atributos modificados:
            - `self.uploaded_ids` (List[int]): IDs dos clientes que enviaram modelos válidos.
            - `self.uploaded_weights` (List[float]): Pesos normalizados baseados na quantidade de amostras.
            - `self.uploaded_models` (List[obj]): Modelos recebidos dos clientes.
        
        Variáveis usadas:
            - `self.client_drop_rate` (float): Fração de clientes que falham em enviar modelo.
            - `self.current_num_join_clients` (int): Número total de clientes que deveriam participar.
            - `self.time_threthold` (float): Limite máximo aceitável de custo de tempo para aceitar modelo.

        Exemplo:
            >>> receive_models()
            # Processa recebimento dos modelos dos clientes ativos dentro do limite de tempo.

        """
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
        Agrega os parâmetros dos modelos enviados pelos clientes para atualizar o modelo global.

        O procedimento é o seguinte:
        - Verifica que existem modelos enviados (`uploaded_models`) para agregar.
        - Cria uma cópia profunda (`deepcopy`) do primeiro modelo enviado para inicializar o modelo global.
        - Zera os valores dos parâmetros do modelo global para preparar a soma ponderada.
        - Itera sobre os pesos normalizados (`uploaded_weights`) e os modelos enviados,
            acumulando os parâmetros ponderados usando o método `add_parameters`.

        Levanta:
            AssertionError: Se a lista `uploaded_models` estiver vazia.

        Dependências:
            - O método `add_parameters(weight, model)` deve estar implementado para somar
            os parâmetros ponderados ao modelo global.

        Efeitos colaterais:
            - Atualiza o atributo `self.global_model` com a agregação dos parâmetros.

        Exemplo:
            >>> aggregate_parameters()
            # Atualiza o modelo global agregando os modelos recebidos dos clientes.

        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        """
        Adiciona os parâmetros ponderados de um modelo cliente ao modelo global.

        Para cada parâmetro correspondente entre o modelo global e o modelo do cliente,
        esta função multiplica o parâmetro do cliente pelo peso `w` e soma ao parâmetro
        do modelo global, atualizando seus valores.

        Args:
            w (float): Peso escalar que indica a importância relativa do modelo do cliente
                    na agregação (por exemplo, proporcional ao número de amostras).
            client_model (torch.nn.Module): Modelo treinado pelo cliente cujos parâmetros
                                            serão agregados ao modelo global.

        Efeitos colaterais:
            - Modifica os parâmetros do modelo global (`self.global_model`), acumulando
            os valores ponderados dos parâmetros do cliente.

        Exemplo:
            >>> add_parameters(0.2, client_model)
            # Adiciona 20% dos parâmetros do client_model ao global_model.

    """
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        """
        Carrega o modelo global pré-treinado a partir do sistema de arquivos.

        O caminho do modelo é construído concatenando o diretório "models", o nome do dataset
        (`self.dataset`) e o nome do arquivo formado pela combinação do algoritmo (`self.algorithm`)
        com o sufixo "_server.pt".

        A função verifica se o arquivo existe e, em caso afirmativo, carrega o modelo usando
        `torch.load` e atribui ao atributo `self.global_model`.

        Levanta:
            AssertionError: Se o arquivo do modelo não existir no caminho esperado.

        Exemplo:
            >>> load_model()
            # Carrega o modelo global do arquivo "models/<dataset>/<algorithm>_server.pt"

        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        """
        Verifica se o arquivo do modelo global pré-treinado existe no sistema de arquivos.

        O caminho do arquivo do modelo é construído a partir do diretório "models", o nome do dataset
        (`self.dataset`) e o nome do arquivo formado pela concatenação do algoritmo (`self.algorithm`)
        com o sufixo "_server.pt".

        Returns:
            bool: True se o arquivo do modelo existir no caminho especificado, False caso contrário.

        Exemplo:
            >>> if model_exists():
            ...     print("Modelo disponível para carregamento")
            ... else:
            ...     print("Modelo não encontrado")

        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        """
        Salva os resultados do experimento em um arquivo HDF5 no diretório "../results/".

        O nome do arquivo é composto pelo dataset, algoritmo, objetivo e número de vezes que o experimento
        foi executado, formando um identificador único.

        O método realiza as seguintes ações:
        - Define o caminho base para salvar os resultados em "../results/".
        - Cria o diretório se ele não existir.
        - Verifica se há dados de acurácia de teste (`self.rs_test_acc`) para salvar.
        - Constrói o nome do arquivo adicionando o dataset, algoritmo, objetivo e número de execuções.
        - Salva os seguintes datasets no arquivo HDF5:
            - `rs_test_acc`: Acurácia dos testes.
            - `rs_test_auc`: AUC (Área sob a curva) dos testes.
            - `rs_train_loss`: Perda durante o treinamento.

        Observação:
        - O arquivo é aberto em modo de escrita, sobrescrevendo qualquer arquivo existente com o mesmo nome.
        - A função imprime o caminho completo do arquivo onde os resultados foram salvos.

        Requisitos:
        - Os atributos `self.rs_test_acc`, `self.rs_test_auc` e `self.rs_train_loss` devem conter os dados
            a serem salvos, geralmente arrays ou listas numéricas.
        - O atributo `self.goal` representa o objetivo do experimento (string).
        - O atributo `self.times` representa o número de vezes que o experimento foi executado (int).

        Exemplo:
            >>> save_results()
            File path: ../results/dataset_algorithm_goal_times.h5
            # Salva os dados em arquivo HDF5

        """
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        """
        Salva um objeto PyTorch (`item`) em disco no formato `.pt` dentro da pasta configurada.

        A função realiza as seguintes etapas:
        - Verifica se o diretório especificado em `self.save_folder_name` existe; se não, cria-o.
        - Salva o objeto `item` no caminho formado por `self.save_folder_name` concatenado com
            o nome do arquivo no formato `"server_<item_name>.pt"`.
        - Utiliza `torch.save` para serializar e salvar o objeto.

        Args:
            item (object): O objeto PyTorch a ser salvo, geralmente um modelo ou estado de otimização.
            item_name (str): Nome identificador do item, usado para formar o nome do arquivo.

        Exemplo:
            >>> save_item(model, "global_model")
            # Salva o modelo em "<save_folder_name>/server_global_model.pt"

        Requisitos:
            - `self.save_folder_name` deve ser uma string válida representando o caminho da pasta para salvar.

        """
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        """
        Carrega um objeto PyTorch previamente salvo a partir do disco.

        O arquivo é buscado dentro do diretório `self.save_folder_name`, com o nome
        no formato `"server_<item_name>.pt"`.

        Args:
            item_name (str): Nome identificador do item a ser carregado, correspondente ao nome usado na função de salvamento.

        Returns:
            object: O objeto PyTorch carregado (ex: modelo, estado de otimizador, etc.).

        Levanta:
            FileNotFoundError: Se o arquivo especificado não existir no caminho esperado.
            Outros erros relacionados à desserialização com `torch.load`.

        Exemplo:
            >>> model = load_item("global_model")
            # Carrega o modelo salvo em "<save_folder_name>/server_global_model.pt"

        Requisitos:
            - `self.save_folder_name` deve estar corretamente definido e conter o arquivo.

        """
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        """
        Calcula métricas de desempenho dos clientes, considerando avaliação diferenciada para novos clientes.

        Comportamento:
        - Se a avaliação de novos clientes (`self.eval_new_clients`) estiver habilitada e houver novos clientes (`self.num_new_clients > 0`),
            realiza fine-tuning para esses novos clientes e retorna suas métricas específicas via
            `fine_tuning_new_clients()` e `test_metrics_new_clients()`.

        - Caso contrário, itera sobre todos os clientes atuais (`self.clients`), coletando as métricas de teste de cada um:
            - `ct`: total de acertos (ex.: número de predições corretas).
            - `ns`: número de amostras testadas pelo cliente.
            - `auc`: área sob a curva ROC do cliente.

        - Acumula:
            - A soma dos acertos por cliente em `tot_correct`.
            - O somatório ponderado do AUC pelo número de amostras em `tot_auc`.
            - O número de amostras em `num_samples`.

        - Coleta os IDs dos clientes em `ids`.

        Returns:
            Tuple[List[int], List[int], List[float], List[float]]:
                - ids: Lista com IDs dos clientes avaliados.
                - num_samples: Lista com o número de amostras testadas por cliente.
                - tot_correct: Lista com a quantidade de acertos por cliente (float).
                - tot_auc: Lista com o AUC ponderado por amostras por cliente (float).

        Requisitos:
            - Cada cliente em `self.clients` deve implementar o método `test_metrics()` que retorna uma tupla `(correct, num_samples, auc)`.

        Exemplo:
            >>> ids, samples, corrects, aucs = test_metrics()
            >>> print(ids)
            [0, 1, 2]
            >>> print(samples)
            [100, 150, 120]
            >>> print(corrects)
            [90.0, 135.0, 108.0]
            >>> print(aucs)
            [85.0, 130.0, 115.0]

        """
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        """
        Calcula métricas de treinamento dos clientes, considerando avaliação diferenciada para novos clientes.

        Comportamento:
        - Se a avaliação de novos clientes (`self.eval_new_clients`) estiver habilitada e houver novos clientes (`self.num_new_clients > 0`),
            retorna listas fixas simulando métricas padrão: IDs `[0]`, número de amostras `[1]` e perda `[0]`.

        - Caso contrário, itera sobre todos os clientes atuais (`self.clients`), coletando as métricas de treinamento de cada um:
            - `cl`: valor da perda (loss) do cliente.
            - `ns`: número de amostras usadas no treinamento pelo cliente.

        - Acumula:
            - O número de amostras em `num_samples`.
            - A perda convertida para float em `losses`.

        - Coleta os IDs dos clientes em `ids`.

        Returns:
            Tuple[List[int], List[int], List[float]]:
                - ids: Lista com IDs dos clientes avaliados.
                - num_samples: Lista com o número de amostras treinadas por cliente.
                - losses: Lista com os valores de perda (loss) por cliente (float).

        Requisitos:
            - Cada cliente em `self.clients` deve implementar o método `train_metrics()` que retorna uma tupla `(loss, num_samples)`.

        Exemplo:
            >>> ids, samples, losses = train_metrics()
            >>> print(ids)
            [0, 1, 2]
            >>> print(samples)
            [100, 150, 120]
            >>> print(losses)
            [0.25, 0.30, 0.20]

        """
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        """
        Avalia o desempenho do modelo agregando métricas de teste e treinamento dos clientes.

        O método realiza as seguintes operações:
        - Obtém as métricas de teste e treinamento dos clientes via `test_metrics()` e `train_metrics()`.
        - Calcula a acurácia média de teste (`test_acc`) e a AUC média de teste (`test_auc`)
            ponderadas pelo número de amostras testadas.
        - Calcula a perda média de treinamento (`train_loss`) ponderada pelo número de amostras treinadas.
        - Calcula as listas de acurácias individuais (`accs`) e AUCs individuais (`aucs`) normalizadas pelo número de amostras.
        - Adiciona a acurácia média de teste e a perda média de treinamento aos arrays `self.rs_test_acc` e `self.rs_train_loss`,
            ou aos arrays opcionais passados via parâmetros `acc` e `loss`.
        - Exibe no console as métricas calculadas, incluindo média e desvio padrão de acurácia e AUC de teste.

        Args:
            acc (list, optional): Lista externa para acumular acurácias médias de teste. Se None,
                                utiliza `self.rs_test_acc`.
            loss (list, optional): Lista externa para acumular perdas médias de treinamento. Se None,
                                utiliza `self.rs_train_loss`.

        Exemplo:
            >>> evaluate()
            Averaged Train Loss: 0.1234
            Averaged Test Accuracy: 0.9123
            Averaged Test AUC: 0.9345
            Std Test Accuracy: 0.0234
            Std Test AUC: 0.0198

        Requisitos:
            - `self.rs_test_acc` e `self.rs_train_loss` devem ser listas existentes para armazenar métricas.
            - Os métodos `test_metrics()` e `train_metrics()` devem estar implementados e retornar métricas compatíveis.

        """
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        """
        Exibe no console as métricas médias de avaliação do modelo.

        Args:
            test_acc (float): Acurácia média obtida na fase de teste.
            test_auc (float): Área sob a curva (AUC) média obtida na fase de teste.
            train_loss (float): Perda média obtida durante o treinamento.

        Exemplo:
            >>> print_(0.9123, 0.9345, 0.1234)
            Average Test Accuracy: 0.9123
            Average Test AUC: 0.9345
            Average Train Loss: 0.1234

        """
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """
        Verifica se critérios de convergência ou parada foram atendidos com base em listas de métricas.

        Esta função avalia listas de sequências de acurácia/perda (`acc_lss`) para determinar
        se as condições de término do treinamento foram satisfeitas, considerando:

        - `top_cnt` (opcional): Número mínimo de iterações desde a melhor métrica (topo) para considerar.
        - `div_value` (opcional): Limite máximo de desvio padrão (divergência) nas últimas `top_cnt` métricas.

        O comportamento depende da combinação dos parâmetros opcionais:
        - Se ambos `top_cnt` e `div_value` forem fornecidos:
            Verifica se a melhor métrica ocorreu há mais de `top_cnt` iterações e se a variabilidade
            das últimas `top_cnt` métricas está abaixo de `div_value`.
        - Se somente `top_cnt` for fornecido:
            Verifica se a melhor métrica ocorreu há mais de `top_cnt` iterações.
        - Se somente `div_value` for fornecido:
            Verifica se a variabilidade das últimas `top_cnt` métricas está abaixo de `div_value`.
        - Se nenhum dos dois for fornecido:
            Levanta `NotImplementedError`.

        Args:
            acc_lss (List[List[float]]): Lista contendo uma ou mais listas de métricas (ex.: acurácias por época).
            top_cnt (int, optional): Número de iterações para considerar no cálculo da posição da melhor métrica
                                    e na análise da variabilidade.
            div_value (float, optional): Valor limite para o desvio padrão das últimas `top_cnt` métricas.

        Returns:
            bool: True se todos os critérios definidos forem satisfeitos para todas as listas em `acc_lss`,
                indicando que o critério de parada foi alcançado; False caso contrário.

        Levanta:
            NotImplementedError: Se nenhum dos parâmetros `top_cnt` ou `div_value` for fornecido.

        Exemplo:
            >>> acc_histories = [[0.8, 0.82, 0.85, 0.86, 0.86], [0.75, 0.78, 0.79, 0.80, 0.81]]
            >>> done = check_done(acc_histories, top_cnt=3, div_value=0.01)
            >>> print(done)
            True

        """
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        """
        Executa o ataque Deep Leakage from Gradients (DLG) para avaliar a privacidade dos modelos enviados pelos clientes.

        Para cada modelo enviado (`uploaded_models`) associado ao seu cliente (`uploaded_ids`), a função:

        - Coloca o modelo em modo avaliação (`eval()`).
        - Calcula o gradiente original como a diferença entre os parâmetros do modelo global e os parâmetros do modelo do cliente.
        - Carrega um subconjunto dos dados de treinamento do cliente usando `load_train_data()`.
        - Para um número limitado de batches (`batch_num_per_client`), coleta os inputs e as saídas do modelo do cliente.
        - Invoca o ataque DLG (função `DLG`) passando o modelo do cliente, o gradiente original e os pares (input, output) coletados.
        - Acumula o valor PSNR retornado pelo ataque, que mede a qualidade da reconstrução do dado, e conta quantos ataques foram realizados com sucesso.

        Ao final:
        - Se pelo menos um ataque DLG foi realizado com sucesso, imprime o valor médio de PSNR em decibéis.
        - Caso contrário, imprime uma mensagem de erro de PSNR.

        Args:
            R (int ou outro tipo): Identificador ou parâmetro para a operação, possivelmente usado para salvar resultados (comentado no código).

        Variáveis usadas:
            - `self.uploaded_ids`: Lista de IDs dos clientes que enviaram modelos.
            - `self.uploaded_models`: Lista dos modelos enviados pelos clientes.
            - `self.global_model`: Modelo global atual para comparação de parâmetros.
            - `self.clients`: Lista ou dicionário de objetos clientes, cada um com método `load_train_data()`.
            - `self.batch_num_per_client`: Número máximo de batches a usar por cliente.
            - `self.device`: Dispositivo PyTorch para movimentação dos tensores (CPU/GPU).

        Observações:
        - O código que salva os itens relacionados ao ataque DLG está comentado.
        - A função `DLG` deve estar definida externamente, recebendo `(model, origin_grad, target_inputs)` e retornando valor PSNR ou None.

        Exemplo:
            >>> call_dlg(1)
            PSNR value is 35.12 dB
            # ou
            PSNR error

        """
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        """
        Inicializa e adiciona novos clientes ao sistema para expansão do conjunto de participantes.

        Para cada novo cliente, cujo ID varia de `self.num_clients` até `self.num_clients + self.num_new_clients - 1`, a função:
        - Lê os dados de treinamento e teste específicos do cliente usando `read_client_data`.
        - Cria uma instância do cliente com a classe fornecida `clientObj`, passando os parâmetros:
            - `id`: identificador único do cliente.
            - `train_samples`: número de amostras de treinamento.
            - `test_samples`: número de amostras de teste.
            - `train_slow` e `send_slow` definidos como False (novos clientes não são lentos por padrão).
        - Adiciona o cliente criado à lista `self.new_clients`.

        Args:
            clientObj (class): Classe do cliente para instanciar os novos clientes. Deve aceitar os parâmetros:
                `args`, `id`, `train_samples`, `test_samples`, `train_slow` e `send_slow`.

        Efeitos colaterais:
            - Modifica o atributo `self.new_clients` adicionando as novas instâncias de clientes.

        Dependências:
            - `self.dataset` deve conter os dados para todos os clientes, inclusive os novos.
            - `self.few_shot` define o parâmetro few-shot para a leitura dos dados.
            - Função `read_client_data(dataset, client_id, is_train, few_shot)` deve estar disponível para leitura dos dados.

        Exemplo:
            >>> set_new_clients(ClientClass)
            # Adiciona novos clientes do tipo ClientClass em self.new_clients

        """
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        """
        Realiza o fine-tuning (ajuste fino) dos modelos dos novos clientes usando os parâmetros do modelo global.

        Para cada cliente em `self.new_clients`, a função:
        - Atualiza os parâmetros do modelo do cliente com os do modelo global atual.
        - Inicializa um otimizador SGD com taxa de aprendizado definida por `self.learning_rate`.
        - Define a função de perda como CrossEntropyLoss.
        - Carrega os dados de treinamento específicos do cliente.
        - Coloca o modelo do cliente em modo treinamento.
        - Executa múltiplas épocas de treinamento (`self.fine_tuning_epoch_new`), iterando sobre batches:
            - Move os dados de entrada e rótulos para o dispositivo do cliente.
            - Calcula a saída do modelo.
            - Calcula a perda.
            - Realiza o passo de backpropagation e atualização dos parâmetros do modelo.

        Requisitos:
        - `self.new_clients` deve conter instâncias de clientes com métodos:
            - `set_parameters()`
            - `load_train_data()`
            - atributo `model` com parâmetros treináveis
            - atributo `device` para movimentação dos tensores
        - `self.learning_rate` deve estar definido para configurar o otimizador.
        - `self.fine_tuning_epoch_new` deve indicar o número de épocas para o fine-tuning.

        Exemplo:
            >>> fine_tuning_new_clients()
            # Ajusta finamente os modelos dos novos clientes com base no modelo global.

        """
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        """
        Calcula as métricas de avaliação dos novos clientes após o fine-tuning.

        Para cada cliente em `self.new_clients`, a função:
        - Obtém as métricas de teste por meio do método `test_metrics()`, que retorna:
            - `ct`: número total de acertos (corretas predições).
            - `ns`: número de amostras testadas.
            - `auc`: área sob a curva ROC.
        - Acumula os valores de acertos convertidos para float em `tot_correct`.
        - Calcula o AUC ponderado pelo número de amostras e acumula em `tot_auc`.
        - Acumula o número de amostras testadas em `num_samples`.
        - Coleta os IDs dos novos clientes em `ids`.

        Returns:
            Tuple[List[int], List[int], List[float], List[float]]:
                - ids: Lista de IDs dos novos clientes avaliados.
                - num_samples: Lista com o número de amostras testadas por cliente.
                - tot_correct: Lista com a quantidade de acertos por cliente (float).
                - tot_auc: Lista com o AUC ponderado por amostras por cliente (float).

        Requisitos:
            - Cada cliente em `self.new_clients` deve implementar o método `test_metrics()`.

        Exemplo:
            >>> ids, samples, corrects, aucs = test_metrics_new_clients()
            >>> print(ids)
            [100, 101]
            >>> print(samples)
            [200, 180]
            >>> print(corrects)
            [180.0, 162.0]
            >>> print(aucs)
            [170.0, 160.0]

        """
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

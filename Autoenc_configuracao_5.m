%% Criando o xtreino 
valores = []; % Aqui entram os valores do banco de dados de treino
 
% Agora vamos precisar fazer as razões e depois trata-las. A ordem é a %seguinte:
% H2 - CH4 - C2H2 - C2H4 - C2H6
 
%Como queremos as razões R2 e R5 de (KIM et al., 2020), temos que 
%R2 = C2H2/C2H4, 
%R5 = C2H4/C2H6. Assim
 
R2 = valores(:,3) ./ valores(:,4); %C2H2/C2H4
R5 = valores(:,4) ./ valores(:,5); %C2H4/C2H6
 
%Agora, precisamos criar o vetor de treino, concatenando as razões
xtreino = horzcat (R2,R5);
 
%E agora precisamos tratar os valores
 
 
%Performamos o seguinte: Para valores > 4, substitui por 4
% Para divisão por zero, substitui por 4
% Para 0/0, substitui por 0
 
%Lidando com valores NaN - Ocorre quando temos 0/0
xtreino(isnan(xtreino)) = 0;
 
for i=1:numel(xtreino)
   if (xtreino(i) == Inf)%Lidando com valores Inf - Ocorre quando temos x/0, um numero qualquer por zero
        xtreino(i) = 4;
    elseif (xtreino(i) > 4) %Lidando com valores acima de 4
      xtreino(i) = 4;
    end
end
 
%Arrumamos o xtreino
xtreino = xtreino';
 
%% Criamos os valores de saída para o treino
xsaida = []; % Aqui entram os valores das respectivas classes dos dados de treinamento
xsaida = xsaida';
 
 
%% Agora vamos repetir os processos para o vetor de testes, Criando xteste
 
valores_teste = []; % Aqui entram os valores do banco de dados de teste
 
% Agora vamos precisar fazer as razões e depois trata-las. A ordem é a seguinte:
% H2 - CH4 - C2H2 - C2H4 - C2H6
 
%Como queremos as razões R2 e R5 de DuvalNew, temos que R2 = C2H2/C2H4, 
%R5 = C2H4/C2H6. Assim
 
R2_teste = valores_teste(:,3) ./ valores_teste(:,4); %C2H2/C2H4
R5_teste = valores_teste(:,4) ./ valores_teste(:,5); %C2H4/C2H6
 
%Agora, precisamos criar o vetor de treino, concatenando as razões
xteste = horzcat (R2_teste,R5_teste);
 
%E agora precisamos tratar os valores
 
 
%Performamos o seguinte: Para valores > 4, substitui por 4
% Para divisão por zero, substitui por 4
% Para 0/0, substitui por 0
 
%Lidando com valores NaN - Ocorre quando temos 0/0
xteste(isnan(xteste)) = 0;
 
for i=1:numel(xteste)
   if (xteste(i) == Inf)%Lidando com valores Inf - Ocorre quando temos x/0, um numero qualquer por zero
        xteste(i) = 4;
    elseif (xteste(i) > 4) %Lidando com valores acima de 4
      xteste(i) = 4;
    end
end
 
%Arrumando xteste
xteste = xteste';
 
 
%% Criando desejado
desejado = []; % Aqui entram os valores das respectivas classes dos dados de teste
 
desejado = desejado';
 
 
%% Criando os vetores no padrão Softmax
 
xtreino_softmax = []; % Aqui entram os valores das respectivas classes dos dados de treinamento, no formato de o’s e 1’s, ex: [00001]
xtreino_softmax = xtreino_softmax';
 
xteste_softmax = []; % Aqui entram os valores das respectivas classes dos dados de teste, no formato de o’s e 1’s, ex: [00001]
xteste_softmax = xteste_softmax';
 
 
%% Criando a Stacked Autoencoder
 

autoenc1 = trainAutoencoder(xtreino,4, ...
    'MaxEpochs',600, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat1 = encode(autoenc1,xtreino);
 
autoenc2 = trainAutoencoder(feat1,5, ...
    'MaxEpochs',600, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1);
 
softnet = trainSoftmaxLayer(feat2,xtreino_softmax,'MaxEpochs',400);
view(softnet)
 
stackednet = stack(autoenc1,autoenc2,softnet);
view(stackednet)
 
%RESULTADO SEM FINE TUNING
%y = stackednet(xteste);
%plotconfusion(xteste_softmax,y);
 
%RESULTADO COM FINE TUNING
stackednet = train(stackednet,xtreino,xtreino_softmax);
y = stackednet(xteste);
plotconfusion(xteste_softmax,y);


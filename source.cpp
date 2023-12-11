#include <cmath>
#include <chrono>
#include <iostream>
#include <cstring>
#include "network.h"

struct dataInfo
{
    double* pixels;
    int digit;
};

dataNetwork ReadDataNetwork(string path)
{
    dataNetwork data{};
    ifstream in;
    in.open(path);
    if (!in.is_open())
    {
        cout << "Error reading file " << path << endl;
        system("pause");
    }
    else
    {
        cout << path << "loading...\n";
    }
    string tmp;
    int L;
    while (!in.eof())
    {
        in >> tmp;
        if (tmp == "Network")
        {
            in >> L;
            data.length = L;
            data.size = new int[L];
            for (int i = 0; i < L; i++)
            {
                in >> data.size[i];
            }
        }
    }
    in.close();
    return data;
}

dataInfo* ReadData(string path, const dataNetwork &dataNW, int &examples)
{
    dataInfo *data;
    ifstream in;
    in.open(path);
    if (!in.is_open())
    {
        cout << "Error reading file " << path << endl;
        system("pause");
    }
    else
    {
        cout << path << " loading...\n";
    }
    string tmp;
    // int L;
    in >> tmp;
    if (std::strcmp("Examples", tmp.data()) == 0)
    {
        in >> examples;
        cout << "Example: " << examples << endl;
        data = new dataInfo[examples];
        for (int i = 0; i < examples; ++i)
        {
            data[i].pixels = new double[dataNW.size[0]];
        }
        for (int i = 0; i < examples; ++i)
        {
            cout << "example " << i << endl;
            in >> data[i].digit;
            cout << "example digit " << data[i].digit << endl;
            for (int j = 0; j < dataNW.size[0]; ++j)
            {
                in >> data[i].pixels[j];
            }
        }
    }
    in.close();
    cout << "lib mnist loaded... \n";
    return data;
}

int main()
{
    Network NW{};
    dataNetwork NW_config;
    dataInfo* datainfo;
    double ra = 0, right, predict, maxra = 0;
    int epoch = 0;
    bool study, repeat = true;
    chrono::duration<double> time;

    NW_config = ReadDataNetwork("Config.txt");
    NW.Init(NW_config);
    NW.PrintConfiguration();

    while (repeat)
    {
        cout << "study 1/0 ";
        cin >> study;
        if (study)
        {
            int examples;
            cout << "Reading file\n";
            datainfo = ReadData("lib_MNIST_edit.txt", NW_config, examples);
            cout << "Reading done\n";
            auto begin = chrono::steady_clock::now();
            while (ra / (double)examples * 100. < 100.)
            {
                cout << "next iteration\n";
                ra = 0;
                auto t1 = chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i)
                {
                    cout << "Current iteration digits: " << datainfo[i].digit << endl;
                    NW.SetInput(datainfo[i].pixels);
                    right = datainfo[i].digit;
                    predict = NW.Forward();
                    if (predict != right)
                    {
                        NW.BackProp(right);
                        NW.UpdateWeights(0.15 * exp(-epoch / 20.));
                    }
                    else
                    {
                        ra++;
                    }
                }
                auto t2 = chrono::steady_clock::now();
                time = t2 - t1;
                if (ra > maxra)
                {
                    maxra = ra;
                }
                cout << "ra: " << ra / (double)examples * 100. << "\t"
                     << "maxra: " << maxra / (double)examples * 100. << endl;
                epoch++;
                cout << "epoch " << epoch << " completed.\n";
                if (epoch == 20)
                {
                    break;
                }
            }
            auto end = chrono::steady_clock::now();
            time = end - begin;
            cout << "TIME: " << time.count() / 60. << " min" << endl;
            NW.SaveWeights();
        }
        else {
            NW.ReadWeights();
        }
        cout << "Test? (1/0)\n";
        bool testflag;
        cin >> testflag;
        if (testflag) {
            int ex_tests;
            dataInfo* dataTest;
            dataTest = ReadData("lib_10k copy.txt", NW_config, ex_tests);
            ra = 0;
            for (int i = 0; i < ex_tests; ++i) {
                NW.SetInput(dataTest[i].pixels);
                predict = NW.Forward();
                right = dataTest[i].digit;
                if (right == predict) {
                    ra++;
                }
            }
            cout << "RA: " << ra / (double)ex_tests * 100. << endl;
        }
        cout << "Repeat? (1/0)\n";
        cin >> repeat; 
    }
    system("pause");
    return 0;
}

def train(model, data_loader, optimizer, device, num_epochs):
    model.train()

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        running_loss = 0.0

        pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (data, _) in enumerate(pbar):
            l_channel, ab_channels = rgb_to_lab(data)
            l_channel, ab_channels = l_channel.to(device), ab_channels.to(device)

            optimizer.zero_grad()

            ab_pred = model(l_channel)

            loss = criterion(ab_pred, ab_channels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{(running_loss / (batch_idx + 1)):.4f}"
            })